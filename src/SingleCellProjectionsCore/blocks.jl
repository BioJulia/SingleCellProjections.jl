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


# TODO: Thread over blocks for sparse matrices
function Base.:*(A::Blocks{T1}, B::Blocks{T2}) where {T1,T2}
	bszA = size(A.blocks)
	bszB = size(B.blocks)
	bszA[2] == bszB[1] || throw(DimensionMismatch("Number of blocks must match for matrix multiplication. Got $bszA and $bszB blocks."))

	# TODO: optimize, we can e.g. use 5-arg mul for dense matrices to avoid copying
	blocks = [sum(k->A.blocks[i,k]*B.blocks[k,j], 1:bszA[2]) for i in 1:bszA[1], j in 1:bszB[2]]

	if length(blocks) == 1
		only(blocks)
	else
		Blocks(blocks)
	end
end



_row_view(A::Matrix{T}, range) where T = @view(A[range, :])


function row_block_view(A::Matrix{T}, heights) where T
	ends = cumsum(heights)
	starts = vcat(1, ends[1:end-1].+1)
	row_ranges = range.(starts, ends)
	blocks = [_row_view(A, r) for r in row_ranges, j in 1:1]
	Blocks(blocks)
end


function Base.:*(A::Blocks{T1}, B::AbstractMatrix{T2}) where {T1,T2}
	hA,wA = size(A)
	hB,wB = size(B)
	wA == hB || (DimensionMismatch("incompatible dimensions for matrix multiplication: tried to multiply a matrix of size ($hA, $wA) with a matrix of size ($hB, $wB). The second dimension of the first matrix: $wA, does not match the first dimension of the second matrix: $hB."))

	heights = size.(A.blocks[1,:], 2)
	# @show heights
	# @show typeof(B)
	BB = row_block_view(B, heights)
	# @show typeof(BB)
	# @info size(BB.blocks)

	A*BB
end




hblock(blocks) = Blocks([block for i in 1:1, block in blocks]) # create 1xN matrix of blocks
vblock(blocks) = Blocks([block for block in blocks, j in 1:1]) # create Nx1 matrix of blocks

