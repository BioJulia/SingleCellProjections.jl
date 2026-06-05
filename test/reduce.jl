using Test
using SingleCellProjections
using ReproducibleJobs: fetch!, forward!
using LinearAlgebra
using DataFrames

function _fix_signs(F::SVD)
	sgn = (sum(F.U.*abs.(F.U); dims=1).>=0).*2 .- 1
	U = F.U .* sgn
	Vt = F.Vt .* sgn'
	SVD(U, F.S, Vt)
end

function run_reduce_tests()
	@testset "Dimension Reductions" begin
		counts_job = Jobs.load_counts(h5_path; sample_names="a")
		counts = fetch!(counts_job)

		counts_sub_job = Jobs.load_counts(h5_subset_path; sample_names="p")

		# TODO: test forwarding
		# TODO: test hash stability

		transformed_job = Jobs.logtransform(counts_job)
		normalized_job = Jobs.normalize_matrix(counts_job)

		@testset "PCA $name" for (name,data_job) in (("logtransformed",transformed_job), ("normalized", normalized_job))
			data = fetch!(data_job)
			var_spec_forwarded = forward!(Jobs.get_var(data_job))
			obs_spec_forwarded = forward!(Jobs.get_obs(data_job))

			X = convert(Matrix, materialize(data.matrix))
			F = _fix_signs(svd(X))

			@testset "nsv=$nsv" for (nsv,rtol) in ((3,1e-1),(10,1e-6))
				Fn = SVD(F.U[:,1:nsv], F.S[1:nsv], F.Vt[1:nsv,:])
				ΣVt_ans = Diagonal(Fn.S)*Fn.Vt

				pca_job = Jobs.pca(data_job; nsv)
				loadings_job = Jobs.loadings(data_job; nsv)

				@test forward!(Jobs.get_obs(pca_job)) == obs_spec_forwarded
				@test forward!(Jobs.get_var(loadings_job)) == var_spec_forwarded

				pca = fetch!(pca_job)
				@test pca.var == DataFrame("PC_id"=>string.("PC",1:nsv))
				test_dataframe_columns_identical("pca.obs vs data.obs", pca.obs, data.obs)
				@test size(pca.matrix) == (nsv, size(data,2))
				@test pca.matrix ≈ ΣVt_ans rtol=rtol

				loadings = fetch!(loadings_job)
				@test loadings.obs == DataFrame("loadings_id"=>string.("loadings",1:nsv))
				test_dataframe_columns_identical("loadings.var vs data.var", loadings.var, data.var)
				@test size(loadings.matrix) == (size(data,1), nsv)
				@test loadings.matrix ≈ Fn.U rtol=rtol

				@testset "projection" begin
					data_sub_job = Jobs.project(data_job, counts_job=>counts_sub_job)
					data_sub = fetch!(data_sub_job)

					pca_sub_job = Jobs.project(pca_job, counts_job=>counts_sub_job)
					@test forward!(Jobs.get_obs(pca_sub_job)) == forward!(Jobs.get_obs(data_sub_job))

					pca_sub = fetch!(pca_sub_job)
					@test pca_sub.var == DataFrame("PC_id"=>string.("PC",1:nsv))
					test_dataframe_columns_identical("pca_sub.obs vs data_sub.obs", pca_sub.obs, data_sub.obs)
					@test size(pca_sub.matrix) == (nsv, size(data_sub, 2))
					@test pca_sub.matrix ≈ ΣVt_ans[:, pbmc_subset_ind] rtol=rtol

					loadings_sub_job = Jobs.project(loadings_job, counts_job=>counts_sub_job)
					@test forward!(loadings_job) === forward!(loadings_job)
					@test fetch!(loadings_job) === fetch!(loadings_job)
				end

			end
		end
	end
end
