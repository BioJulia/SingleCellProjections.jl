@testset "Transforms" begin
	counts_job = Jobs.load_counts(h5_path; sample_names="a")
	counts = fetch!(counts_job)

	counts_sub_job = Jobs.load_counts(h5_subset_path; sample_names="p")

	# TODO: test forwarding
	# TODO: test hash stability

	@testset "logtransform scale_factor=$scale_factor T=$T" for scale_factor in (10_000, 1_000), T in (Float64,Float32)
		X = T.(simple_logtransform(expected_mat, scale_factor))

		T_args = T==Float64 ? () : (T,)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)
		l_job = Jobs.logtransform(T_args..., counts_job; kwargs...)

		@test forward(Jobs.get_obs(l_job)).spec == forward(Jobs.get_obs(counts_job)).spec
		@test forward(Jobs.get_var(l_job)).spec == forward(Jobs.get_var(counts_job)).spec

		let l = fetch!(l_job)
			@test l.matrix.matrix ≈ X
			@test nnz(l.matrix.matrix) == expected_nnz
			@test eltype(l.matrix.matrix) == T

			test_dataframe_columns_identical("l.var vs counts.var", l.var, counts.var)
			test_dataframe_columns_identical("l.obs vs counts.obs", l.obs, counts.obs)
		end

		p_job = Jobs.project(l_job, counts_job=>counts_sub_job)

		# Test that it is identical to logtransforming the matrix without projection
		p_matrix_job = Jobs.get_matrix(p_job)
		p_matrix_job2 = Jobs.get_matrix(Jobs.logtransform(T_args..., counts_sub_job; kwargs...))
		@test forward(p_matrix_job).spec == forward(p_matrix_job2).spec

		# Vars shouldn't be changed by the projection
		@test forward(Jobs.get_var(p_job)).spec == forward(Jobs.get_var(counts_sub_job)).spec

		# Obs shouldn't be changed by the projection
		@test forward(Jobs.get_obs(p_job)).spec == forward(Jobs.get_obs(counts_sub_job)).spec


		let p = fetch!(p_job), Xs = sparse(X[:,pbmc_subset_ind]), counts_sub = fetch!(counts_sub_job)
			@test p.matrix.matrix ≈ Xs
			@test nnz(p.matrix.matrix) == nnz(Xs)
			@test eltype(p.matrix.matrix) == T

			test_dataframe_columns_identical("p.var vs counts_sub.var", p.var, counts_sub.var)
			test_dataframe_columns_identical("p.obs vs counts_sub.obs", p.obs, counts_sub.obs)
		end


		# Variable subsetting
		@testset "var_filter" begin
			var_mask = expected_feature_names .> "L"
			X_filtered = T.(simple_logtransform(expected_mat[var_mask,:], scale_factor))

			l_filtered_job = Jobs.logtransform(T_args..., counts_job; var_filter="name"=>>("L"), kwargs...)

			@test forward(Jobs.get_obs(l_filtered_job)).spec == forward(Jobs.get_obs(counts_job)).spec

			let l_filtered = fetch!(l_filtered_job)
				@test l_filtered.matrix.matrix ≈ X_filtered
				@test eltype(l_filtered.matrix.matrix) == T

				@test isequal(l_filtered.var, expected_var[var_mask,:])
				test_dataframe_columns_identical("l_filtered.obs vs counts.obs", l_filtered.obs, counts.obs)
			end

			p_filtered_job = Jobs.project(l_filtered_job, counts_job=>counts_sub_job)


			# Test that it is identical to logtransforming the matrix without projection
			p_filtered_matrix_job = Jobs.get_matrix(p_filtered_job)
			p_filtered_matrix_job2 = Jobs.get_matrix(Jobs.logtransform(T_args..., counts_sub_job; var_filter="name"=>>("L"), kwargs...))
			@test forward(p_filtered_matrix_job).spec == forward(p_filtered_matrix_job2).spec

			# Vars where subsetted. Alternatively to testing the result, we could've tested the spec.
			@test isequal(fetch!(Jobs.get_var(p_filtered_job)), fetch!(Jobs.get_var(counts_sub_job))[var_mask,:])

			# Obs shouldn't be changed by the projection
			@test forward(Jobs.get_obs(p_filtered_job)).spec == forward(Jobs.get_obs(counts_sub_job)).spec


			let p_filtered = fetch!(p_filtered_job), Xs = sparse(X_filtered[:, pbmc_subset_ind]), counts_sub = fetch!(counts_sub_job)
				@test p_filtered.matrix.matrix ≈ Xs
				@test nnz(p_filtered.matrix.matrix) == nnz(Xs)
				@test eltype(p_filtered.matrix.matrix) == T

				@test isequal(p_filtered.var, counts_sub.var[var_mask,:])
				test_dataframe_columns_identical("p_filtered.obs vs counts_sub.obs", p_filtered.obs, counts_sub.obs)
			end


			# TODO: Projection where the projected counts have partially different genes
		end
	end

end
