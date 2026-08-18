
# split into one vector for each instead?
struct BarnesHutNode{N1,N2}
	child_lengths::NTuple{N2,Int}
	center_of_gravity::SVector{N1,Float64}
	skip_pointer::Int
end

BarnesHutNode(N1,N2) = BarnesHutNode{N1,N2}(ntuple(i->0,N2), (@SVector zeros(N1)), 0)


# this is a quadtree/octree/etc with additional info stored
mutable struct BarnesHutTree{N1,N2} # make immutable?
	max_depth::Int
	leaf_size::Int
	mins::SVector{N1,Float64}
	maxes::SVector{N1,Float64}
	point_indices::Vector{Int}
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






function buildrec!(tree::BarnesHutTree{N1,N2}, points::AbstractVector, point_range::UnitRange{Int}, mins::SVector{N1,Float64}, maxes::SVector{N1,Float64}, depth::Int) where {N1,N2}
	mid = (mins+maxes)/2

	push!(tree.nodes, BarnesHutNode(N1,N2)) # dummy initialization to reserve space
	this_node_ind = length(tree.nodes)

	# setup scratch spaces
	point_indices = tree.scratch1
	resize!(point_indices, length(point_range))
	point_indices .= view(tree.point_indices,point_range)

	child_scratch = tree.scratch2
	resize!(child_scratch,N2)
	child_scratch .= 0

	child_ids = tree.scratch3
	resize!(child_ids, length(point_indices))

	for (i,i2) in enumerate(point_indices)
		bucket_id = childind(points[i2],mid)
		child_ids[i] = bucket_id
		child_scratch[bucket_id] += 1
	end
	child_lengths = ntuple(i->child_scratch[i], N2)

	w = first(point_range)
	for i=1:N2 # for each bucket
		child_length = child_scratch[i]
		child_scratch[i] = w
		w += child_length
	end
	@assert w == last(point_range)+1

	for (i,bucket_ind) in enumerate(child_ids)
		w = child_scratch[bucket_ind]
		tree.point_indices[w] = point_indices[i]
		child_scratch[bucket_ind] = w+1
	end



	point_sum = @SVector zeros(N1) # used to compute center_of_gravity

	# recurse
	if depth<tree.max_depth
		k1 = first(point_range)
		for i=1:N2 # for each child
			k2 = k1 + child_lengths[i]
			child_range = k1:k2-1

			if length(child_range)>tree.leaf_size
				child_mins  = SVector(ntuple(j->((i-1)&(1<<(j-1))==0 ? mins[j] : mid[j]  ), N1))
				child_maxes = SVector(ntuple(j->((i-1)&(1<<(j-1))==0 ? mid[j]  : maxes[j]), N1))
				point_sum += buildrec!(tree, points, child_range, child_mins, child_maxes, depth+1)
			elseif !isempty(child_range)
				point_sum += sum(p->points[tree.point_indices[p]], child_range)
			end

			k1 = k2
		end
	else
		point_sum = sum(p->points[tree.point_indices[p]], point_range)
	end

	center_of_gravity = point_sum/max(1,length(point_range)) # max to avoid div by zero
	tree.nodes[this_node_ind] = BarnesHutNode(child_lengths, center_of_gravity, length(tree.nodes)+1)

	point_sum
end


function build!(tree::BarnesHutTree{N1,N2}, points::AbstractVector; max_depth::Int=20, leaf_size::Int=10) where {N1,N2}
	nbr_points = length(points)

	# clear existing tree, but reuse memory
	resize!(tree.point_indices, nbr_points)
	tree.point_indices .= 1:nbr_points
	empty!(tree.nodes)

	# set params
	tree.max_depth = max_depth
	tree.leaf_size = leaf_size

	# compute bounding box
	boundingbox!(tree, points)

	if !isempty(points)
		@assert N1==length(points[1])
		buildrec!(tree, points, 1:nbr_points, tree.mins, tree.maxes, 0)
	end
	tree
end

