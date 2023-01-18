function validate_barnes_hut_tree_rec(tree::BarnesHutTree{N1,N2}, points, depth, nodeInd, mins, maxes, pointRange) where {N1,N2}
	node = tree.nodes[nodeInd]
	@assert node.centerOfGravity ≈ mean(points[tree.pointIndices[pointRange]])
	mid = (mins+maxes)/2

	@assert all(>=(0), node.childLengths)
	@assert sum(node.childLengths) == length(pointRange)

	# for each child
	firstPointInd = first(pointRange)
	childNodeInd = nodeInd+1
	for (childInd,cartesianInd) in enumerate(CartesianIndices(ntuple(i->2,N1)))
		childMask = Tuple(cartesianInd) .== 2 # false/true for each dimension
		childMins  = SVector(ntuple(i->childMask[i] ? mid[i]   : mins[i], N1))
		childMaxes = SVector(ntuple(i->childMask[i] ? maxes[i] : mid[i] , N1))

		endPointInd = firstPointInd + node.childLengths[childInd]
		childRange = firstPointInd:endPointInd-1

		# check that all child points are within the child node
		for p in points[tree.pointIndices[childRange]]
			@assert all(p.>=childMins)
			@assert all(p.<=childMaxes)
		end

		if depth<tree.maxDepth && length(childRange)>tree.leafSize
			validate_barnes_hut_tree_rec(tree, points, depth+1, childNodeInd, childMins, childMaxes, childRange)
			childNodeInd = tree.nodes[childNodeInd].skipPointer
		end

		firstPointInd = endPointInd
	end
	@assert firstPointInd-1 == last(pointRange)
end

function validate_barnes_hut_tree(tree, points)
	for p in points
		@assert all(p.>=tree.mins)
		@assert all(p.<=tree.maxes)
	end
	validate_barnes_hut_tree_rec(tree, points, 0, 1, tree.mins, tree.maxes, 1:length(points))
end

@testset "barnes_hut" begin

@testset "Basic $(d)d npoints=$N" for d=2:3, N in (4,20,80,10000)
	try
		rng = StableRNG(2014)
		points = randn(rng, SVector{d,Float64}, N)

		tree = BarnesHutTree(d)
		build!(tree, points; leafSize=2)
		validate_barnes_hut_tree(tree, points)
		@test true
	catch err
		# A little hack to stop at the first error in the @testset
		if err isa AssertionError
			showerror(stdout, err, catch_backtrace())
			@test false
		else
			rethrow(err)
		end
	end
end


end
