using Test
using SingleCellProjections
using ReproducibleJobs: fetch!, forward!
using SCTransform
using SparseArrays
using DataFrames

function run_transform_tests()
	@testset "Transforms" begin
		counts_job = SCP.load_counts(h5_path; sample_names="a")
		counts = fetch!(counts_job)

		counts_sub_job = SCP.load_counts(h5_subset_path; sample_names="p")

		# TODO: test forwarding
		# TODO: test hash stability

		@testset "logtransform scale_factor=$scale_factor T=$T" for scale_factor in (10_000, 1_000), T in (Float64,Float32)
			T_args = T==Float64 ? () : (T,)
			kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)

			@testset "standard" begin
				X = T.(simple_logtransform(expected_mat, scale_factor))

				l_job = SCP.logtransform(T_args..., counts_job; kwargs...)

				@test forward!(SCP.get_obs(l_job)) == forward!(SCP.get_obs(counts_job))
				@test forward!(SCP.get_var(l_job)) == forward!(SCP.get_var(counts_job))

				let l = fetch!(l_job)
					@test unblockify(l.matrix) ≈ X
					@test nnz(unblockify(l.matrix)) == expected_nnz
					@test eltype(unblockify(l.matrix)) == T

					test_dataframe_columns_identical("l.var vs counts.var", l.var, counts.var)
					test_dataframe_columns_identical("l.obs vs counts.obs", l.obs, counts.obs)
				end

				p_job = SCP.project(l_job, counts_job=>counts_sub_job)

				# Test that it is identical to logtransforming the matrix without projection
				p_matrix_job = SCP.get_matrix(p_job)
				p_matrix_job2 = SCP.get_matrix(SCP.logtransform(T_args..., counts_sub_job; kwargs...))
				@test forward!(p_matrix_job) == forward!(p_matrix_job2)

				# Vars shouldn't be changed by the projection
				@test forward!(SCP.get_var(p_job)) == forward!(SCP.get_var(counts_sub_job))

				# Obs shouldn't be changed by the projection
				@test forward!(SCP.get_obs(p_job)) == forward!(SCP.get_obs(counts_sub_job))


				let p = fetch!(p_job), Xs = sparse(X[:,pbmc_subset_ind]), counts_sub = fetch!(counts_sub_job)
					@test unblockify(p.matrix) ≈ Xs
					@test nnz(unblockify(p.matrix)) == nnz(Xs)
					@test eltype(unblockify(p.matrix)) == T

					test_dataframe_columns_identical("p.var vs counts_sub.var", p.var, counts_sub.var)
					test_dataframe_columns_identical("p.obs vs counts_sub.obs", p.obs, counts_sub.obs)
				end
			end

			# With matrix Blocks, we shouldn't need var_filter support for logtransform. Just use filtering before instead. So we remove these tests at least for now.
			# # Variable subsetting
			# @testset "var_filter" begin
			# 	var_mask = expected_feature_names .> "L"
			# 	X = T.(simple_logtransform(expected_mat[var_mask,:], scale_factor))

			# 	l_job = SCP.logtransform(T_args..., counts_job; var_filter="name"=>>("L"), kwargs...)

			# 	@test forward!(SCP.get_obs(l_job)) == forward!(SCP.get_obs(counts_job))

			# 	let l = fetch!(l_job)
			# 		@test unblockify(l.matrix) ≈ X
			# 		@test eltype(unblockify(l.matrix)) == T

			# 		@test isequal(l.var, expected_var[var_mask,:])
			# 		test_dataframe_columns_identical("l.obs vs counts.obs", l.obs, counts.obs)
			# 	end

			# 	p_job = SCP.project(l_job, counts_job=>counts_sub_job)


			# 	# Test that it is identical to logtransforming the matrix without projection
			# 	p_matrix_job = SCP.get_matrix(p_job)
			# 	p_matrix_job2 = SCP.get_matrix(SCP.logtransform(T_args..., counts_sub_job; var_filter="name"=>>("L"), kwargs...))
			# 	@test forward!(p_matrix_job) == forward!(p_matrix_job2)

			# 	# Vars where subsetted. Alternatively to testing the result, we could've tested the spec.
			# 	@test isequal(fetch!(SCP.get_var(p_job)), fetch!(SCP.get_var(counts_sub_job))[var_mask,:])

			# 	# Obs shouldn't be changed by the projection
			# 	@test forward!(SCP.get_obs(p_job)) == forward!(SCP.get_obs(counts_sub_job))


			# 	let p = fetch!(p_job), Xs = sparse(X[:, pbmc_subset_ind]), counts_sub = fetch!(counts_sub_job)
			# 		@test unblockify(p.matrix) ≈ Xs
			# 		@test nnz(unblockify(p.matrix)) == nnz(Xs)
			# 		@test eltype(unblockify(p.matrix)) == T

			# 		@test isequal(p.var, counts_sub.var[var_mask,:])
			# 		test_dataframe_columns_identical("p.obs vs counts_sub.obs", p.obs, counts_sub.obs)
			# 	end


			# 	# TODO: Projection where the projected counts have partially different genes
			# end
		end


		@testset "sctransform T=$T annotate=$annotate" for T in (Float64,Float32), annotate in (false,true)
			T_args = T==Float64 ? () : (T,)
			kwargs = annotate ? (; annotate) : (;)

			@testset "standard" begin
				X = sctransform(expected_sparse, counts.var, params)
				sct_job = SCP.sctransform(T_args..., counts_job; kwargs...)

				@test forward!(SCP.get_obs(sct_job)) == forward!(SCP.get_obs(counts_job))
				# TODO: test var forwarding?
				# @show forward!(SCP.get_var(sct_job))

				let sct = fetch!(sct_job)
					# var
					cols = ["id", "name", "feature_type", "genome"]
					@test sct.var.id == params.id
					@test sct.var.name == params.name
					@test sct.var.feature_type == params.feature_type
					@test all(==("GRCh38"), sct.var.genome)
					if annotate
						push!(cols, "logGeneMean", "outlier", "beta0", "beta1", "theta")
						@test sct.var.logGeneMean ≈ params.logGeneMean
						@test sct.var.outlier == params.outlier
						@test sct.var.beta0 ≈ params.beta0
						@test sct.var.beta1 ≈ params.beta1
						@test sct.var.theta ≈ params.theta
					end
					@test names(sct.var) == cols

					# obs
					test_dataframe_columns_identical("sct.obs vs counts.obs", sct.obs, counts.obs)

					# matrix
					@test size(sct.matrix) == size(X)
					@test eltype(sct.matrix.terms[1].matrix) == T
					@test materialize(sct.matrix) ≈ X rtol=1e-3
				end

				p_job = SCP.project(sct_job, counts_job=>counts_sub_job)

				@test forward!(SCP.get_obs(p_job)) == forward!(SCP.get_obs(counts_sub_job))
				# TODO: test var forwarding?
				# @show forward!(SCP.get_var(p_job))

				let p = fetch!(p_job), counts_sub = fetch!(counts_sub_job)
					Xs = sctransform(expected_sparse[:, pbmc_subset_ind], counts.var, params; clip=sqrt(size(X,2)/30))

					# var
					cols = ["id", "name", "feature_type", "genome"]
					@test p.var.id == params.id
					@test p.var.name == params.name
					@test p.var.feature_type == params.feature_type
					@test all(==("GRCh38"), p.var.genome)
					if annotate
						push!(cols, "logGeneMean", "outlier", "beta0", "beta1", "theta")
						@test p.var.logGeneMean ≈ params.logGeneMean
						@test p.var.outlier == params.outlier
						@test p.var.beta0 ≈ params.beta0
						@test p.var.beta1 ≈ params.beta1
						@test p.var.theta ≈ params.theta
					end
					@test names(p.var) == cols

					# obs
					test_dataframe_columns_identical("p.obs vs counts_sub.obs", p.obs, counts_sub.obs)

					# matrix
					@test size(p.matrix) == size(Xs)
					@test eltype(p.matrix.terms[1].matrix) == T
					@test materialize(p.matrix) ≈ Xs rtol=1e-3
				end
			end

			@testset "var_filter" begin
				var_mask = expected_feature_names .> "C"
				es = expected_sparse[var_mask,:]
				params_filtered = scparams(es, DataFrame(id=expected_feature_ids, name=expected_feature_names, feature_type=expected_feature_types)[var_mask,:]; use_cache=false)
				X = sctransform(es, counts.var[var_mask,:], params_filtered)

				sct_job = SCP.sctransform(T_args..., counts_job; var_filter="name"=>.>("C"), kwargs...)

				@test forward!(SCP.get_obs(sct_job)) == forward!(SCP.get_obs(counts_job))

				# TODO: test var forwarding?
				# @show forward!(SCP.get_var(sct_job))

				let sct = fetch!(sct_job)
					# var
					cols = ["id", "name", "feature_type", "genome"]
					@test sct.var.id == params_filtered.id
					@test sct.var.name == params_filtered.name
					@test sct.var.feature_type == params_filtered.feature_type
					@test all(==("GRCh38"), sct.var.genome)
					if annotate
						push!(cols, "logGeneMean", "outlier", "beta0", "beta1", "theta")
						@test sct.var.logGeneMean ≈ params_filtered.logGeneMean
						@test sct.var.outlier == params_filtered.outlier
						@test sct.var.beta0 ≈ params_filtered.beta0
						@test sct.var.beta1 ≈ params_filtered.beta1
						@test sct.var.theta ≈ params_filtered.theta
					end
					@test names(sct.var) == cols

					# obs
					test_dataframe_columns_identical("sct.obs vs counts.obs", sct.obs, counts.obs)

					# matrix
					@test size(sct.matrix) == size(X)
					@test eltype(sct.matrix.terms[1].matrix) == T
					@test materialize(sct.matrix) ≈ X rtol=1e-3
				end


				p_job = SCP.project(sct_job, counts_job=>counts_sub_job)

				@test forward!(SCP.get_obs(p_job)) == forward!(SCP.get_obs(counts_sub_job))
				# TODO: test var forwarding?
				# @show forward!(SCP.get_var(p_job))

				let p = fetch!(p_job), counts_sub = fetch!(counts_sub_job)
					Xs = sctransform(es[:, pbmc_subset_ind], counts.var[var_mask,:], params_filtered; clip=sqrt(size(X,2)/30))

					# var
					cols = ["id", "name", "feature_type", "genome"]
					@test p.var.id == params_filtered.id
					@test p.var.name == params_filtered.name
					@test p.var.feature_type == params_filtered.feature_type
					@test all(==("GRCh38"), p.var.genome)
					if annotate
						push!(cols, "logGeneMean", "outlier", "beta0", "beta1", "theta")
						@test p.var.logGeneMean ≈ params_filtered.logGeneMean
						@test p.var.outlier == params_filtered.outlier
						@test p.var.beta0 ≈ params_filtered.beta0
						@test p.var.beta1 ≈ params_filtered.beta1
						@test p.var.theta ≈ params_filtered.theta
					end
					@test names(p.var) == cols

					# obs
					test_dataframe_columns_identical("p.obs vs counts_sub.obs", p.obs, counts_sub.obs)

					# matrix
					@test size(p.matrix) == size(Xs)
					@test eltype(p.matrix.terms[1].matrix) == T
					@test materialize(p.matrix) ≈ Xs rtol=1e-3
				end
			end
		end


	end
end
