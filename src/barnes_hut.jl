
# split into one vector for each instead?
struct BarnesHutNode{N1,N2}
	childLengths::NTuple{N2,Int}
	centerOfGravity::SVector{N1,Float64}
	skipPointer::Int
end

BarnesHutNode(N1,N2) = BarnesHutNode{N1,N2}(ntuple(i->0,N2), (@SVector zeros(N1)), 0)


# this is a quadtree/octree/etc with additional info stored
mutable struct BarnesHutTree{N1,N2} # make immutable?
	maxDepth::Int
	leafSize::Int
	mins::SVector{N1,Float64}
	maxes::SVector{N1,Float64}
	pointIndices::Vector{Int}
	nodes::Vector{BarnesHutNode{N1,N2}}
	# scratch spaces
	scratch1::Vector{Int}
	scratch2::Vector{Int}
	scratch3::Vector{Int}
end
BarnesHutTree(ndim::Int) = BarnesHutTree{ndim,2^ndim}(0, 0, (@SVector zeros(ndim)), (@SVector zeros(ndim)), [], [], [], [], [])

function boundingbox!(tree::BarnesHutTree{N1,N2}, points::AbstractVector) where {N1,N2}
	mins  = Inf*@SVector ones(N1)
	maxes = -Inf*@SVector ones(N1)

	for p in points
		mins  = min.(mins,  p)
		maxes = max.(maxes, p)
	end
	tree.mins  = mins
	tree.maxes = maxes
	nothing
end

diameter2(tree::BarnesHutTree) = sum(abs2,tree.maxes-tree.mins)



# Map a point to a value in 1:2^d
# What's the best way to write this function using SVectors?
childind(p::SVector{N,T}, mid::SVector{N,T}) where {N,T} = 1+sum((p.>=mid).*SVector(ntuple(i->2^(i-1),N)))






function buildrec!(tree::BarnesHutTree{N1,N2}, points::AbstractVector, pointRange::UnitRange{Int}, mins::SVector{N1,Float64}, maxes::SVector{N1,Float64}, depth::Int) where {N1,N2}
	mid = (mins+maxes)/2

	push!(tree.nodes, BarnesHutNode(N1,N2)) # dummy initialization to reserve space
	thisNodeInd = length(tree.nodes)

	# setup scratch spaces
	pointIndices = tree.scratch1
	resize!(pointIndices, length(pointRange))
	pointIndices .= view(tree.pointIndices,pointRange)

	childScratch = tree.scratch2
	resize!(childScratch,N2)
	childScratch .= 0

	childIds = tree.scratch3
	resize!(childIds, length(pointIndices))

	for (i,i2) in enumerate(pointIndices)
		bucketId = childind(points[i2],mid)
		childIds[i] = bucketId
		childScratch[bucketId] += 1
	end
	childLengths = ntuple(i->childScratch[i], N2)

	w = first(pointRange)
	for i=1:N2 # for each bucket
		childLength = childScratch[i]
		childScratch[i] = w
		w += childLength
	end
	@assert w == last(pointRange)+1

	for (i,bucketInd) in enumerate(childIds)
		w = childScratch[bucketInd]
		tree.pointIndices[w] = pointIndices[i]
		childScratch[bucketInd] = w+1
	end



	pointSum = @SVector zeros(N1) # used to compute centerOfGravity

	# recurse
	if depth<tree.maxDepth
		k1 = first(pointRange)
		for i=1:N2 # for each child
			k2 = k1 + childLengths[i]
			childRange = k1:k2-1

			if length(childRange)>tree.leafSize
				childMins  = SVector(ntuple(j->((i-1)&(1<<(j-1))==0 ? mins[j] : mid[j]  ), N1))
				childMaxes = SVector(ntuple(j->((i-1)&(1<<(j-1))==0 ? mid[j]  : maxes[j]), N1))
				pointSum += buildrec!(tree, points, childRange, childMins, childMaxes, depth+1)
			elseif !isempty(childRange)
				pointSum += sum(p->points[tree.pointIndices[p]], childRange)
			end

			k1 = k2
		end
	else
		pointSum = sum(p->points[tree.pointIndices[p]], pointRange)
	end

	centerOfGravity = pointSum/max(1,length(pointRange)) # max to avoid div by zero
	tree.nodes[thisNodeInd] = BarnesHutNode(childLengths, centerOfGravity, length(tree.nodes)+1)

	pointSum
end


function build!(tree::BarnesHutTree{N1,N2}, points::AbstractVector; maxDepth::Int=20, leafSize::Int=10) where {N1,N2}
	nbrPoints = length(points)

	# clear existing tree, but reuse memory
	resize!(tree.pointIndices, nbrPoints)
	tree.pointIndices .= 1:nbrPoints
	empty!(tree.nodes)

	# set params
	tree.maxDepth = maxDepth
	tree.leafSize = leafSize

	# compute bounding box
	boundingbox!(tree, points)

	if !isempty(points)
		@assert N1==length(points[1])
		buildrec!(tree, points, 1:nbrPoints, tree.mins, tree.maxes, 0)
	end
	tree
end

