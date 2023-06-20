function process_rows(f,(X,offset))
	for j in 1:size(X,2)
		f(X, j, j+offset)
	end
end

function threaded_sparse_row_worker(f,channel)
	while true
		item = take!(channel)
		isnothing(item) && break # no more chunks to process
		process_rows(f,item)
		# yield() # Shouldn't be needed with take! above
	end
end

function threaded_sparse_row_map(f, X::AbstractSparseMatrix{Tv,Ti};
                                 chunk_size=100,
                                 nworkers=Threads.nthreads()-1,
                                 channel_size=nworkers*4,
                                ) where {Tv<:Real,Ti<:Integer}
	nworkers = max(nworkers,1)
	P,N = size(X)

	local channel
	local workers

	if nworkers>1
		channel = Channel{Union{Nothing,Tuple{SparseMatrixCSC{Tv,Ti},Int}}}(channel_size)
		workers = map(1:nworkers) do _
			Threads.@spawn threaded_sparse_row_worker(f, channel)
			# @async threaded_sparse_row_worker(f, channel) # TEMP
		end
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
		offset = first(row_range)-1

		if nworkers>1
			put!(channel, (chunk,offset))
		else
			process_rows(f, (chunk,offset))
		end
	end

	# Tell workers to stop
	if nworkers>1
		for i in 1:nworkers
			put!(channel, nothing)
		end
		wait.(workers)
	end

	nothing
end
