@testset "Sum Squared" begin
	counts_job = Jobs.load_counts(h5_path; sample_names="a")
	normalized_job = Jobs.normalize_matrix(Jobs.sctransform(counts_job))

	# TODO: projections
	# TODO: test forwarding
	# TODO: test hash stability

	# Skip counts unless we find a way to handle automatic centering, but only if needed...
	# @testset "variance and std $name" for (name, data_job) in (("counts", counts_job), ("normalized", normalized_job))
	@testset "variance and std $name" for (name, data_job) in (("normalized", normalized_job),)
		data = fetch!(data_job)
		X = convert(Matrix{Float64}, unblockify(materialize(data)))
		P, N = size(data)
		id_col = only(names(data.var,1))

		expected_var = vec(var(X; dims=2))
		expected_std = vec(std(X; dims=2))

		@testset "variance" begin
			v_job = Jobs.variance(data_job)
			v = fetch!(v_job)
			@test v isa DataFrame
			@test names(v) == [id_col, "variance"]
			@test isequal(v[!, id_col], data.var[!, id_col])
			@test v.variance ≈ expected_var
		end

		@testset "variance col kwarg" begin
			v_job = Jobs.variance(data_job; col="my_var")
			v = fetch!(v_job)
			@test names(v) == [id_col, "my_var"]
			@test v.my_var ≈ expected_var
		end

		@testset "std" begin
			s_job = Jobs.std(data_job)
			s = fetch!(s_job)
			@test s isa DataFrame
			@test names(s) == [id_col, "std"]
			@test isequal(s[!, id_col], data.var[!, id_col])
			@test s.std ≈ expected_std
		end

		@testset "std col kwarg" begin
			s_job = Jobs.std(data_job; col="my_std")
			s = fetch!(s_job)
			@test names(s) == [id_col, "my_std"]
			@test s.my_std ≈ expected_std
		end
	end
end
