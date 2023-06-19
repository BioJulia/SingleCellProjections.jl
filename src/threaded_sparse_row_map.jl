function threaded_sparse_row_worker(f,channel)
	while true
		item = take!(channel)
		isnothing(item) && break # no more chunks to process

		X,offset = item
		for j in 1:size(X,2)
			f(X, j, j+offset)
		end
		# yield() # Shouldn't be needed with take! above
	end
end

function threaded_sparse_row_map(f, X::AbstractSparseMatrix{Tv,Ti};
                                 chunk_size=100,
                                 nthreads=Threads.nthreads(),
                                 channel_size=nthreads*4,
                                ) where {Tv<:Real,Ti<:Integer}
	nthreads = max(nthreads,1)
	P,N = size(X)

	channel = Channel{Union{Nothing,Tuple{SparseMatrixCSC{Tv,Ti},Int}}}(channel_size)

	workers = map(1:nthreads) do _
		Threads.@spawn threaded_sparse_row_worker(f, channel)
	end

	colptr_curr = first.(nzrange.(Ref(X),1:N))
	colptr_end = last.(nzrange.(Ref(X),1:N))
	rowval = rowvals(X)
	nzval = nonzeros(X)

	rowval_scratch = Vector{Ti}() # will grow but get reused between chunks
	nzval_scratch  = Vector{Tv}() # will grow but get reused between chunks

	for row_range in Iterators.partition(1:P, chunk_size)
		colptr_chunk = Vector{Ti}(undef, N+1)

		for j in 1:N
			colptr_chunk[j] = length(rowval_scratch)+1

			c = colptr_curr[j]

			while c<=colptr_end[j] && rowval[c]<=last(row_range)
				push!(rowval_scratch, rowval[c] - first(row_range) + 1)
				push!(nzval_scratch, nzval[c])
				c += 1
			end

			colptr_curr[j] = c
		end
		colptr_chunk[end] = length(rowval_scratch)+1

		rowval_chunk = copy(rowval_scratch)
		nzval_chunk = copy(nzval_scratch)

		empty!(rowval_scratch)
		empty!(nzval_scratch)

		chunk = SparseMatrixCSC(length(row_range), N, colptr_chunk, rowval_chunk, nzval_chunk)
		chunk = permutedims(chunk,(2,1)) # transpose

		put!(channel, (chunk,first(row_range)-1))
	end

	# Tell workers to stop
	for i in 1:nthreads
		put!(channel, nothing)
	end

	wait.(workers)

	nothing
end
