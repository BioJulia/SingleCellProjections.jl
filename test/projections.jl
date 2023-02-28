@testset "Projections" begin
	proj_obs_ind = 1:12:size(counts,2)
	counts_proj = counts[:,proj_obs_ind]

	@testset "from" begin
		@test_throws ArgumentError project(counts_proj,transformed)
		t2 = project(counts_proj, transformed; from=counts)
		@test materialize(t2) ≈ materialize(transformed)[:,proj_obs_ind] rtol=1e-3

		l2 = logtransform(counts_proj)
		@test_throws ArgumentError project(l2,normalized)
		n2 = project(l2,normalized; from=transformed)
		@test size(n2) == (size(normalized,1),size(l2,2)) # TODO: test the result more properly?
	end

	@testset "models" begin
		fl = force_layout(reduced; ndim=3, k=10, rng=StableRNG(408))
		fl_proj = project(counts_proj, fl.models)
		@test materialize(fl_proj)≈materialize(fl)[:,proj_obs_ind] rtol=1e-5
	end


	# TODO: project with different set of variables

end
