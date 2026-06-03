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

		@testset "nsv=$nsv" for (nsv,rtol) in ((3,1e-1),(10,1e-6))
			F = svd(X)
			F = SVD(F.U[:,1:nsv], F.S[1:nsv], F.Vt[1:nsv,:])
			pca_job = Jobs.pca(data_job; nsv)

			@test forward!(Jobs.get_obs(pca_job)) == obs_spec_forwarded

			let pca = fetch!(pca_job)
				# var
				@test pca.var == DataFrame("PC_id"=>string.("PC",1:nsv))

				# obs
				test_dataframe_columns_identical("pca.obs vs data.obs", pca.obs, data.obs)

				# matrix
				ΣVt = pca.matrix
				@test size(ΣVt) == (nsv, size(data,2))

				ΣVt_ans = Diagonal(F.S)*F.Vt
				sign_fix = sign.(sum(ΣVt .* ΣVt_ans; dims=2))
				ΣVt_ans .*= sign_fix

				@test ΣVt ≈ ΣVt_ans rtol=rtol
			end
		end
	end


end
end
