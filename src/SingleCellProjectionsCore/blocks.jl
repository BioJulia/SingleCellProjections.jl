function _validated_size(sizes::Matrix{Tuple{Int,Int}})
	# Compute sizes and ensure they are consistent
	hs = sum(first, sizes; dims=1)
	ws = sum(last, sizes; dims=2)
	(allequal(hs) && allequal(ws)) || throw(DimensionMismatch("Block sizes do not match: $sizes"))
	(first(hs), first(ws))
end

# Or use BlockMatrix from BlockArrays.jl?
struct Blocks{T}
	blocks::Matrix{T} # Make immutable? Use tuples somehow?
	sz::Tuple{Int,Int}

	function Blocks(blocks::Matrix{T}) where T
		isconcretetype(T) || throw(ArgumentError("Only heterogeneous block matrices are currently allowed."))
		new{T}(blocks, _validated_size(size.(blocks)))
	end
end


Base.size(b::Blocks) = b.sz
Base.size(b::Blocks, i::Int) = i in (1,2) ? b.sz[i] : 1

function Base.convert(::Type{Matrix}, b::Blocks)
	hvcat(size(b.blocks,2), permutedims(b.blocks)...)
end
function Base.convert(::Type{Matrix{T}}, b::Blocks) where T
	hvcat(size(b.blocks,2), convert.(Matrix{T}, permutedims(b.blocks))...)
end


Base.adjoint(A::Blocks) = Blocks([adjoint(A.blocks[j,i]) for i in 1:size(A.blocks,2), j in 1:size(A.blocks,1)])
Base.transpose(A::Blocks) = Blocks([transpose(A.blocks[j,i]) for i in 1:size(A.blocks,2), j in 1:size(A.blocks,1)])


# TODO: Share code between these two.
function Base.copyto!(dest::Matrix{D}, src::Blocks{T}) where {D,T}
	col_start = 1
	for j in 1:size(src.blocks, 2)
		col_end = col_start + size(src.blocks[1,j], 2) - 1
		col_range = col_start:col_end

		row_start = 1
		for i in 1:size(src.blocks, 1)
			block = src.blocks[i,j]

			row_end = row_start + size(block, 1) - 1
			row_range = row_start:row_end

			dst = @view(dest[row_range, col_range])
			copyto!(dst, block)

			row_start = row_end+1
		end
		col_start = col_end+1
	end
end
function MatrixExpressions.addto!(dest, src::Blocks{T}) where T
	col_start = 1
	for j in 1:size(src.blocks, 2)
		col_end = col_start + size(src.blocks[1,j], 2) - 1
		col_range = col_start:col_end

		row_start = 1
		for i in 1:size(src.blocks, 1)
			block = src.blocks[i,j]

			row_end = row_start + size(block, 1) - 1
			row_range = row_start:row_end

			dst = @view(dest[row_range, col_range])
			MatrixExpressions.addto!(dst, block)

			row_start = row_end+1
		end
		col_start = col_end+1
	end
end


function block_sizes_to_ranges(s)
	ends = cumsum(s)
	starts = vcat(1, ends[1:end-1].+1)
	range.(starts, ends)
end
get_row_ranges(A::Blocks{T}) where T = block_sizes_to_ranges(size.(@view(A.blocks[:,1]), 1))
get_col_ranges(A::Blocks{T}) where T = block_sizes_to_ranges(size.(@view(A.blocks[1,:]), 2))





# This will be removed later when we support heterogeneous blocks
MatrixExpressions._is_dense(A::Blocks{T}) where T = MatrixExpressions._is_dense(A.blocks[1,1])
MatrixExpressions._is_sparse(A::Blocks{T}) where T = MatrixExpressions._is_sparse(A.blocks[1,1])
MatrixExpressions._is_adjoint_type(A::Blocks{T}) where T = MatrixExpressions._is_adjoint_type(T)

# TODO: Figure out where to put this code
function MatrixExpressions._pnz(A::Blocks{T}) where T
	if MatrixExpressions._is_sparse(A)
		sum(nnz, A.blocks) / prod(size(A))
	else
		-1.0 # -1 means dense
	end
end
MatrixExpressions.MatrixInfo(A::Blocks{T}) where T = MatrixExpressions.MatrixInfo(size(A), T, MatrixExpressions._pnz(A))





function blockify(A::AbstractMatrix; row_block_size, col_block_size)
	row_ranges = ChunkSplitters.chunks(1:size(A,1); size=row_block_size)
	col_ranges = ChunkSplitters.chunks(1:size(A,2); size=col_block_size)

	Blocks([A[rr,cr] for rr in row_ranges, cr in col_ranges])
end

# function _block_scalar_product(a::AbstractVector{T1}, b::AbstractVector{T2}) where {T1,T2}
# 	sum(k->a[k]*b[k], 1:length(a))
# end



struct SummationTree{T}
	nodes::Vector{Union{Nothing,T}} # nodes[1] is the root. Children of node i are (2i,2i+1)
	# remaining::Vector{Threads.Atomic{Int}}

	n_children::Vector{Int} # The number of children for each node. We might not need this later, but it makes the algorithm easier to write.
	n_remaining::AtomicMemory{Int}
end
function SummationTree(::Type{T}, n_leaves::Int) where T
	n_leaves_full = nextpow(2,n_leaves) # The number of leaves if the we have a full tree (i.e. the number of leaves is 2^n)
	n_internal = n_leaves_full-1
	n_nodes  = n_internal + n_leaves
	nodes = Vector{Union{Nothing,T}}(nothing, n_nodes)
	n_remaining = AtomicMemory{Int}(undef, n_internal)


	n_children = zeros(Int, n_internal)
	for i in n_internal:-1:1
		has_left = get(n_children, 2i, 2i<=n_nodes) > 0
		has_right = get(n_children, 2i+1, 2i+1<=n_nodes) > 0
		nc = has_left + has_right
		n_children[i] = nc
		@atomic n_remaining[i] = nc
	end
	SummationTree(nodes, n_children, n_remaining)
end

# must be called exactly once per leaf
function add_result!(tree::SummationTree{T}, leaf_i::Int, result::T) where T
	n_nodes = length(tree.nodes)
	n_internal = length(tree.n_remaining)
	n_leaves = n_nodes - n_internal

	@assert 1 <= leaf_i <= n_leaves
	i = leaf_i + n_internal
	tree.nodes[i] = result

	i == 1 && return # degenerate case with exactly one result

	parent_i = div(i,2)
	nr = @atomic tree.n_remaining[parent_i] -= 1
	while nr == 0 # we can propagate the result upward
		if tree.n_children[parent_i] == 1 # no summation needed, just move the result upward (happens due to unbalanced tree)
			tree.nodes[parent_i] = tree.nodes[i]
			tree.nodes[i] = nothing
		else
			tree.nodes[parent_i] = tree.nodes[2parent_i] # init with left child
			MatrixExpressions.addto!(tree.nodes[parent_i], tree.nodes[2parent_i+1]) # add right child
			tree.nodes[2parent_i] = nothing
			tree.nodes[2parent_i+1] = nothing
		end

		parent_i == 1 && break # the full tree has been summed

		i = parent_i
		parent_i = div(i,2)
		nr = @atomic tree.n_remaining[parent_i] -= 1
	end
end

# Can only be called after add_result! has been called for each leaf
get_result(tree::SummationTree{T}) where T = tree.nodes[1]::T



# TODO: Thread over blocks for sparse matrices
function Base.:*(A::Blocks{T1}, B::Blocks{T2}) where {T1,T2}
	Ni,Nk = size(A.blocks)
	Nk2,Nj = size(B.blocks)
	Nk == Nk2 || throw(DimensionMismatch("Number of blocks must match for matrix multiplication. Got $((Ni,Nk)) and $((Nk2,Nj)) blocks."))

	# bszA = size(A.blocks)
	# bszB = size(B.blocks)
	# bszA[2] == bszB[1] || throw(DimensionMismatch("Number of blocks must match for matrix multiplication. Got $bszA and $bszB blocks."))

	# TODO: optimize, we can e.g. use 5-arg mul for dense matrices to avoid copying
	# blocks = [sum(k->A.blocks[i,k]*B.blocks[k,j], 1:bszA[2]) for i in 1:bszA[1], j in 1:bszB[2]]
	# blocks = [_block_scalar_product(@view(A.blocks[i,:]), @view(B.blocks[:,j])) for i in 1:bszA[1], j in 1:bszB[2]]


	# @show T1
	# @show T2
	# @show MatrixExpressions._is_dense_type(T1)
	# @show MatrixExpressions._is_dense_type(T2)


	if MatrixExpressions._is_dense_type(T1) && MatrixExpressions._is_dense_type(T2)
		# Threaded by BLAS - loop over blocks and use mul!
		# @info "Dense × Dense"

		# TODO: implement properly
		blocks = [sum(k->A.blocks[i,k]*B.blocks[k,j], 1:Nk) for i in 1:Ni, j in 1:Nj]
	elseif MatrixExpressions._is_dense_type(T1) || MatrixExpressions._is_dense_type(T2)
		# Not threaded by BLAS, output is dense, thread over blocks and combine pairwise

		# @info "Dense × Sparse or Sparse × Dense"

		# TODO: implement properly
		# blocks = [sum(k->A.blocks[i,k]*B.blocks[k,j], 1:Nk) for i in 1:Ni, j in 1:Nj]

		T_out = Base.promote_op(*, eltype(T1), eltype(T2))
		T_out = Base.promote_op(+, T_out, T_out)
		trees = [SummationTree(Matrix{T_out}, Nk) for i in 1:Ni, j in 1:Nj]

		# # TODO: Thread it!
		# for j in 1:Nj
		# 	for i in 1:Ni
		# 		for k in 1:Nk
		# 			# @info (j,k,i)
		# 			add_result!(trees[i,j], k, A.blocks[i,k]*B.blocks[k,j])
		# 		end
		# 	end
		# end

		# Basic threading
		tforeach(CartesianIndices((Nk,Ni,Nj))) do c # TODO: Configure OhMyThreads scheduler
		# foreach(CartesianIndices((Nk,Ni,Nj))) do c # DEBUG version without threading
			(k,i,j) = Tuple(c)
			# @info "$(Threads.threadid()): ($i,$j,$k)"

			# @show typeof(A.blocks[i,k]), typeof(B.blocks[k,j])
			# @show typeof(A.blocks[i,k] * B.blocks[k,j])
			# @show typeof(trees[i,j])

			add_result!(trees[i,j], k, A.blocks[i,k]*B.blocks[k,j])


			# @info "mul"
			# @time C = A.blocks[i,k]*B.blocks[k,j]

			# # TESTING
			# a = A.blocks[i,k]
			# b = B.blocks[k,j]
			# @time C = Matrix{Float64}(undef, size(a,1), size(b,2))
			# @time mul!(C, a, b)

			# @info "add_result"
			# @time add_result!(trees[i,j], k, C)
		end

		blocks = [get_result(trees[i,j]) for i in 1:Ni, j in 1:Nj]
	else
		# Not threaded by BLAS, output is sparse(/something else?), thread over blocks combine all at once
		# @info "Sparse × Sparse"

		# TODO: implement properly
		blocks = [sum(k->A.blocks[i,k]*B.blocks[k,j], 1:Nk) for i in 1:Ni, j in 1:Nj]
	end


	if length(blocks) == 1
		only(blocks)
	else
		Blocks(blocks)
	end
end






_col_view(A::Matrix{T}, range) where T = @view(A[:, range])
_row_view(A::Matrix{T}, range) where T = @view(A[range, :])


# TODO: revise this solution - there must a better way to handle Diagonals
function _row_view(D::Diagonal{E,T}, range) where {E,T}
	d = D.diag
	sparse(1:length(range), range, d[range], length(range), length(d))
end


_row_view(A::Adjoint{E,T}, range) where {E,T} = _col_view(A', range)'
_col_view(A::Adjoint{E,T}, range) where {E,T} = _row_view(A', range)'
_row_view(A::Transpose{E,T}, range) where {E,T} = transpose(_col_view(transpose(A), range))
_col_view(A::Transpose{E,T}, range) where {E,T} = transpose(_row_view(transpose(A), range))


function row_block_view(A::AbstractMatrix{T}, heights) where T
	# ends = cumsum(heights)
	# starts = vcat(1, ends[1:end-1].+1)
	# row_ranges = range.(starts, ends)
	row_ranges = block_sizes_to_ranges(heights)
	blocks = [_row_view(A, r) for r in row_ranges, j in 1:1]
	Blocks(blocks)
end


# function reblock(A::Blocks{T}; sparse_chunksize=400) where T<:SparseMatrixCSC
# 	nblocks = 0
# 	changed = false
# 	for j in 1:size(A.blocks,2)
# 		nb = cld(size(A.blocks[1,j],2), sparse_chunksize)
# 		nblocks += nb
# 		changed = changed || nb != 1
# 	end
# 	changed || return A

# 	blocks = Matrix{T}(undef, size(A.blocks,1), nblocks) # TODO: FIX - DO ABSOLUTELY NOT CONVERT TO T

# 	# @show size(A.blocks)

# 	for i in 1:size(A.blocks,1)
# 		j2 = 1
# 		for j in 1:size(A.blocks,2)
# 			curr = A.blocks[i,j]
# 			# @show i,j
# 			# @show size(curr)
# 			nb = cld(size(curr,2), sparse_chunksize)
# 			for k in 1:nb
# 				# @show i,j2
# 				blocks[i,j2] = @view curr[:,(k-1)*sparse_chunksize+1:min(k*sparse_chunksize,end)]
# 				j2 += 1
# 			end
# 		end
# 	end

# 	Blocks(blocks)
# end


function reblock(A::Blocks{T}; sparse_chunksize=512) where T<:SparseMatrixCSC
	block_info = Tuple{Int,UnitRange{Int}}[]
	for j in 1:size(A.blocks,2)
		append!(block_info, tuple.(j, ChunkSplitters.chunks(1:size(A.blocks[1,j],2); size=sparse_chunksize)))
	end
	first.(block_info) == 1:size(A.blocks,2) && return A

	blocks = [@view(A.blocks[i,j][:,r]) for i in 1:size(A.blocks,1), (j,r) in block_info]
	Blocks(blocks)
end

reblock(A::Blocks{<:Adjoint{<:Any,<:SparseMatrixCSC}}; kwargs...) = reblock(A'; kwargs...)' # could avoid some allocations, but probably doesn't matter in the end


# reblock(A::AbstractMatrix; kwargs...) = reblock(Blocks(fill(A,1,1)); kwargs...)



function Base.:*(A::Blocks{T1}, B::AbstractMatrix{T2}) where {T1,T2}
	hA,wA = size(A)
	hB,wB = size(B)
	wA == hB || (DimensionMismatch("incompatible dimensions for matrix multiplication: tried to multiply a matrix of size ($hA, $wA) with a matrix of size ($hB, $wB). The second dimension of the first matrix: $wA, does not match the first dimension of the second matrix: $hB."))

	# heights = size.(A.blocks[1,:], 2)
	# BB = row_block_view(B, heights)
	# A*BB

	AA = MatrixExpressions._is_sparse_type(T1) ? reblock(A) : A
	heights = size.(@view(AA.blocks[1,:]), 2)
	BB = row_block_view(B, heights)
	AA*BB
end



# TODO: Generalize these to handle Blocks as inputs
hblock(blocks) = Blocks([block for i in 1:1, block in blocks]) # create 1xN matrix of blocks
# vblock(blocks) = Blocks([block for block in blocks, j in 1:1]) # create Nx1 matrix of blocks


function hblock(a::AbstractVector{<:Blocks})
	# n_row_blocks = size(first(a), 1)
	# n_col_blocks = sum(x->size(x,2), a)
	# @assert all(x->size(x,1)==n_row_blocks, a)
	Blocks(reduce(hcat, getfield.(a, :blocks)))
end






# Since we have our own Blocks type, we need to implement the chunking to get SCTransform.jl to work
function SCTransform.gene_chunk_producer(channel, A::Blocks{<:SparseMatrixCSC{Tv,Ti}};
                                         feature_mask,
                                         chunk_size=128) where {Tv,Ti}
	row_ranges = get_row_ranges(A)

	for i in 1:size(A.blocks,1)
		hc = hcat(@view(A.blocks[i,:])...)
		t = copy(transpose(hc))

		n = size(t,2)
		for gene_chunk in ChunkSplitters.chunks(1:n; size=chunk_size)
			chunk = @view t[:,gene_chunk]
			feature_offset = first(row_ranges[i])-1 + first(gene_chunk)-1
			put!(channel, (chunk,feature_offset))
		end
	end
end


