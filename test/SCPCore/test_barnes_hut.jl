function validate_barnes_hut_tree_rec(tree::BarnesHutTree{N1,N2}, points, depth, node_ind, mins, maxes, point_range) where {N1,N2}
	node = tree.nodes[node_ind]
	@assert node.center_of_gravity ≈ mean(points[tree.point_indices[point_range]])
	mid = (mins+maxes)/2

	@assert all(>=(0), node.child_lengths)
	@assert sum(node.child_lengths) == length(point_range)

	# for each child
	first_point_ind = first(point_range)
	child_node_ind = node_ind+1
	for (child_ind,cartesian_ind) in enumerate(CartesianIndices(ntuple(i->2,N1)))
		child_mask = Tuple(cartesian_ind) .== 2 # false/true for each dimension
		child_mins  = SVector(ntuple(i->child_mask[i] ? mid[i]   : mins[i], N1))
		child_maxes = SVector(ntuple(i->child_mask[i] ? maxes[i] : mid[i] , N1))

		end_point_ind = first_point_ind + node.child_lengths[child_ind]
		child_range = first_point_ind:end_point_ind-1

		# check that all child points are within the child node
		for p in points[tree.point_indices[child_range]]
			@assert all(p.>=child_mins)
			@assert all(p.<=child_maxes)
		end

		if depth<tree.max_depth && length(child_range)>tree.leaf_size
			validate_barnes_hut_tree_rec(tree, points, depth+1, child_node_ind, child_mins, child_maxes, child_range)
			child_node_ind = tree.nodes[child_node_ind].skip_pointer
		end

		first_point_ind = end_point_ind
	end
	@assert first_point_ind-1 == last(point_range)
end

function validate_barnes_hut_tree(tree, points)
	for p in points
		@assert all(p.>=tree.mins)
		@assert all(p.<=tree.maxes)
	end
	validate_barnes_hut_tree_rec(tree, points, 0, 1, tree.mins, tree.maxes, 1:length(points))
end

function run_barnes_hut_tests()
	@testset "barnes_hut" begin
		@testset "Basic $(d)d npoints=$N" for d=2:3, N in (4,20,80,10000)
			try
				rng = StableRNG(2014)
				points = randn(rng, SVector{d,Float64}, N)

				tree = BarnesHutTree(d)
				build!(tree, points; leaf_size=2)
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
end
