@testset "Subsetting" begin
	P,N = (50,587)

	counts_job = Jobs.load_counts(h5_path; sample_names="a")

	# TODO: projections
	# TODO: test forwarding
	# TODO: test hash stability

	# TODO: Test for more data matrices
	# @testset "subset $name" for (name,data_job) in (("counts",counts_job), ("normalized",normalized_job), ("reduced",reduced_job))
	@testset "subset $name" for (name,data_job) in (("counts",counts_job),)
		data = fetch!(data_job)
		data_spec_forwarded = forward(data_job).spec
		var_spec_forwarded = forward(Jobs.get_var(data_job)).spec
		obs_spec_forwarded = forward(Jobs.get_obs(data_job)).spec

		X = materialize(data)
		P,N = size(data)

		s_job = Jobs.subset_var(data_job, select(data.var,:id))
		@test forward(s_job).spec == data_spec_forwarded
		let s = fetch!(s_job)
			@test materialize(s) ≈ X
			test_dataframe_columns_identical("s.var vs data.var", s.var, data.var)
			test_dataframe_columns_identical("s.obs vs data.obs", s.obs, data.obs)
		end

		s_job = Jobs.subset_obs(data_job, select(data.obs,:cell_id))
		@test forward(s_job).spec == data_spec_forwarded
		let s = fetch!(s_job)
			@test materialize(s) ≈ X
			test_dataframe_columns_identical("s.var vs data.var", s.var, data.var)
			test_dataframe_columns_identical("s.obs vs data.obs", s.obs, data.obs)
		end

		s_job = Jobs.subset_matrix(data_job, select(data.var,:id), select(data.obs,:cell_id))
		@test forward(s_job).spec == data_spec_forwarded
		let s = fetch!(s_job)
			@test materialize(s) ≈ X
			test_dataframe_columns_identical("s.var vs data.var", s.var, data.var)
			test_dataframe_columns_identical("s.obs vs data.obs", s.obs, data.obs)
		end


		var_ind_subset = P:-2:1
		obs_ind_subset = N:-5:1


		var_ref_job = ReproducibleJobs.Job(SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=collect(var_ind_subset)))
		obs_ref_job = ReproducibleJobs.Job(SingleCellProjections.create_datamatrix_getindex_spec(data_job; obs_ind=collect(obs_ind_subset)))
		matrix_ref_job = ReproducibleJobs.Job(SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=collect(var_ind_subset), obs_ind=collect(obs_ind_subset)))


		var_ids_subset = select(data.var,:id)[var_ind_subset, :]
		obs_ids_subset = select(data.obs,:cell_id)[obs_ind_subset, :]

		s_job = Jobs.subset_var(data_job, var_ids_subset)
		@test forward(Jobs.get_obs(s_job)).spec == obs_spec_forwarded
		@test forward(s_job).spec == forward(var_ref_job).spec
		let s = fetch!(s_job)
			@test materialize(s) ≈ X[var_ind_subset, :]
			@test isequal(s.var, data.var[var_ind_subset, :])
			test_dataframe_columns_identical("s.obs vs data.obs", s.obs, data.obs)
		end

		s_job = Jobs.subset_obs(data_job, obs_ids_subset)
		@test forward(Jobs.get_var(s_job)).spec == var_spec_forwarded
		@test forward(s_job).spec == forward(obs_ref_job).spec
		let s = fetch!(s_job)
			@test materialize(s) ≈ X[:, obs_ind_subset]
			test_dataframe_columns_identical("s.var vs data.var", s.var, data.var)
			@test isequal(s.obs, data.obs[obs_ind_subset, :])
		end

		s_job = Jobs.subset_matrix(data_job, var_ids_subset, obs_ids_subset)
		@test forward(s_job).spec == forward(matrix_ref_job).spec
		let s = fetch!(s_job)
			@test materialize(s) ≈ X[var_ind_subset, obs_ind_subset]
			@test isequal(s.var, data.var[var_ind_subset, :])
			@test isequal(s.obs, data.obs[obs_ind_subset, :])
		end


		var_ids_bad = vcat(var_ids_subset[1:2,:], DataFrame("id"=>["not_an_id"]))
		obs_ids_bad = vcat(obs_ids_subset[1:2,:], DataFrame("cell_id"=>["not_an_id"]))
		@test_throws "not_an_id" fetch!(Jobs.subset_var(data_job, var_ids_bad))
		@test_throws "not_an_id" fetch!(Jobs.subset_obs(data_job, obs_ids_bad))
		@test_throws "not_an_id" fetch!(Jobs.subset_matrix(data_job, var_ids_bad, obs_ids_bad))

		@test_throws "Column names didn't match" fetch!(Jobs.subset_var(data_job, obs_ids_subset))
		@test_throws "Column names didn't match" fetch!(Jobs.subset_obs(data_job, var_ids_subset))
		@test_throws "Column names didn't match" fetch!(Jobs.subset_matrix(data_job, obs_ids_subset, var_ids_subset))
	end
end
