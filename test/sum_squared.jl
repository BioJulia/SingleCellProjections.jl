function relative_std_ref(X; kwargs...)
	s = vec(std(X; kwargs...));
	s ./ maximum(s)
end

@testset "Sum Squared" begin
	counts_job = Jobs.load_counts(h5_path; sample_names="a")
	normalized_job = Jobs.normalize_matrix(Jobs.sctransform(counts_job))

	counts_sub_job = Jobs.load_counts(h5_subset_path; sample_names="p")
	normalized_sub_job = Jobs.project(normalized_job, counts_job => counts_sub_job)

	# TODO: test forwarding
	# TODO: test hash stability

	# Skip counts unless we find a way to handle automatic centering, but only if needed...
	# @testset "variance and std $name" for (name, data_job) in (("counts", counts_job), ("normalized", normalized_job))
	@testset "variance and std $name" for (name, data_job) in (("normalized", normalized_job),)
		data = fetch!(data_job)
		X = convert(Matrix{Float64}, unblockify(materialize(data)))
		P, N = size(data)
		id_col = only(names(data.var,1))

		@testset "$col" for (f, g, col) in (
				(Jobs.variance,     var,              "variance"),
				(Jobs.std,          std,              "std"),
				(Jobs.relative_std, relative_std_ref, "relative_std"))
			expected = vec(g(X; dims=2))
			job = f(data_job)
			result = fetch!(job)
			@test result isa DataFrame
			@test names(result) == [id_col, col]
			@test isequal(result[!, id_col], data.var[!, id_col])
			@test result[!, col] ≈ expected

			job2 = f(data_job; col="my_$col")
			result2 = fetch!(job2)
			@test names(result2) == [id_col, "my_$col"]
			@test result2[!, "my_$col"] ≈ expected
		end
	end

	@testset "projections" begin
		data_base = fetch!(normalized_job)
		data_proj = fetch!(normalized_sub_job)
		X_proj = convert(Matrix{Float64}, unblockify(materialize(data_proj)))
		id_col = only(names(data_base.var, 1))

		# NB: mean=0 because centering is already done using the mean from the base data
		@testset "$col projections" for (f, g, col) in (
				(Jobs.variance,     var,              "variance"),
				(Jobs.std,          std,              "std"),
				(Jobs.relative_std, relative_std_ref, "relative_std"))
			# project=:no (default): uses base data regardless of projection
			base_job = f(normalized_job)
			proj_job = Jobs.project(base_job), counts_job => counts_sub_job)
			@test isequal(forward!(base_job), forward!(proj_job))

			v = Jobs.value_column_data(base_job)
			v_p = Jobs.project(v)
			@test isequal(forward!(v), forward!(v_p))

			base = fetch!(base_job)
			proj = fetch!(proj_job)
			@test names(proj) == [id_col, col]
			@test proj[!, col] ≈ base[!, col]

			# project=:yes: uses projected data
			proj_yes_job = Jobs.project(f(normalized_job; project=:yes), normalized_job => normalized_sub_job)
			result_yes = fetch!(proj_yes_job)
			@test names(result_yes) == [id_col, col]
			@test result_yes[!, col] ≈ vec(g(X_proj; mean=zeros(size(X_proj,1),1), dims=2))
		end
	end
end
