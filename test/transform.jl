@testset "Transforms" begin
	counts_job = Jobs.load_counts(h5_path; sample_names="a")
	counts = fetch!(counts_job)

	# TODO: projections
	# TODO: test forwarding
	# TODO: test hash stability

	@testset "logtransform scale_factor=$scale_factor T=$T" for scale_factor in (10_000, 1_000), T in (Float64,Float32)
		X = T.(simple_logtransform(expected_mat, scale_factor))
		args = T==Float64 ? (counts_job,) : (T, counts_job)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)
		l_job = Jobs.logtransform(args...; kwargs...)


		@test forward(Jobs.get_obs(l_job)).spec == forward(Jobs.get_obs(counts_job)).spec
		@test forward(Jobs.get_var(l_job)).spec == forward(Jobs.get_var(counts_job)).spec

		let l = fetch!(l_job)
			@test l.matrix.matrix ≈ X
			@test nnz(l.matrix.matrix) == expected_nnz
			@test eltype(l.matrix.matrix) == T

			test_dataframe_columns_identical("l.var vs counts.var", l.var, counts.var)
			test_dataframe_columns_identical("l.obs vs counts.obs", l.obs, counts.obs)
		end

		# Variable subsetting
		@testset "var_filter" begin
			var_mask = expected_feature_names .> "L"
			X_filtered = T.(simple_logtransform(expected_mat[var_mask,:], scale_factor))

			l_filtered_job = Jobs.logtransform(args...; var_filter="name"=>>("L"), kwargs...)

			@test forward(Jobs.get_obs(l_filtered_job)).spec == forward(Jobs.get_obs(counts_job)).spec

			let l_filtered = fetch!(l_filtered_job)
				@test l_filtered.matrix.matrix ≈ X_filtered
				@test eltype(l_filtered.matrix.matrix) == T

				@test isequal(l_filtered.var, expected_var[var_mask,:])
				test_dataframe_columns_identical("l_filtered.obs vs counts.obs", l_filtered.obs, counts.obs)
			end
		end
	end

end
