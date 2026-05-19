@testset "Filtering" begin
	P,N = (50,587)

	counts_job = Jobs.load_counts(h5_path; sample_names="a")
	counts_job = Jobs.add_obs_column(counts_job, "group", counts_obs_group)
	counts_job = Jobs.add_obs_column(counts_job, "value", counts_obs_value)


	var_annot_df = select(fetch!(Jobs.get_var(counts_job)), ["id", "name"])[end:-4:1, :]
	var_annot_spec = SingleCellProjections.table_getindex_spec(Jobs.annotation(Jobs.get_var(counts_job), "name"), P:-4:1)

	obs_annot_df = select(fetch!(Jobs.get_obs(counts_job)), ["cell_id", "barcode"])[end:-5:1, :]
	obs_annot_spec = SingleCellProjections.table_getindex_spec(Jobs.annotation(Jobs.get_obs(counts_job), "barcode"), N:-5:1)


	# TODO: projections
	# TODO: test forwarding
	# TODO: test hash stability

	# TODO: Test for more data matrices
	# @testset "filter $name" for (name,data_job) in (("counts",counts_job), ("normalized",normalized_job), ("reduced",reduced_job))
	@testset "filter $name" for (name,data_job) in (("counts",counts_job),)
		data = fetch!(data_job)
		data_spec_forwarded = forward!(data_job)
		var_spec_forwarded = forward!(Jobs.get_var(data_job))
		obs_spec_forwarded = forward!(Jobs.get_obs(data_job))

		X = unblockify(materialize(data))
		P,N = size(data)

		f_job = Jobs.filter_matrix(:, :, data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X
			test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
			test_dataframe_columns_identical("f.obs vs data.obs", f.obs, data.obs)
		end
		@test forward!(f_job) == data_spec_forwarded

		f_job = Jobs.filter_var(1:2:P, data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[1:2:P, :]
			@test isequal(f.var, data.var[1:2:P,:])
			test_dataframe_columns_identical("f.obs vs data.obs", f.obs, data.obs)
		end
		@test forward!(Jobs.get_obs(f_job)) == obs_spec_forwarded
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=1:2:P)
		@test forward!(f_job) == forward!(ref_job)
		# NB: Projection should fail unless the var IDs are identical!

		f_job = Jobs.filter_obs(1:2:N, data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[:, 1:2:N]
			test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
			@test isequal(f.obs, data.obs[1:2:N,:])
		end
		@test forward!(Jobs.get_var(f_job)) == var_spec_forwarded
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; obs_ind=1:2:N)
		@test forward!(f_job) == forward!(ref_job)
		# NB: Projection should fail unless the obs IDs are identical!

		f_job = Jobs.filter_matrix(1:3:P, 1:10:N, data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[1:3:P, 1:10:N]
			@test isequal(f.var, data.var[1:3:P,:])
			@test isequal(f.obs, data.obs[1:10:N,:])
		end
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=1:3:P, obs_ind=1:10:N)
		@test forward!(f_job) == forward!(ref_job)
		# NB: Projection should fail unless the var and obs IDs are identical!

		f_job = Jobs.filter_obs("group"=>==("A"), data_job)
		@test forward!(Jobs.get_var(f_job)) == var_spec_forwarded
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[:, data.obs.group.=="A"]
			test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
			@test isequal(f.obs, filter("group"=>==("A"), data.obs))
		end
		ref_obs_ind = findall(==("A"), data.obs.group)
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; obs_ind=ref_obs_ind)
		@test forward!(f_job) == forward!(ref_job)

		# Hmm. How do we support this? The problem is the anonymous function that cannot be hashed currently. Implement HashableFunctions?
		# f_job = Jobs.filter_obs(["group","value"]=>(g,v)->g=="A" && v<1.0, data_job)
		# let f = fetch!(f_job)
		# 	@test unblockify(materialize(f)) ≈ X[:, (data.obs.group.=="A") .& (data.obs.value.<1.0)]
		# 	test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
		# 	@test isequal(f.obs, filter(["group","value"]=>(g,v)->g=="A" && v<1.0, data.obs))
		# end


		f_job = Jobs.filter_matrix(1:3:P, "group"=>==("A"), data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[1:3:P, data.obs.group.=="A"]
			@test isequal(f.var, data.var[1:3:P,:])
			@test isequal(f.obs, data.obs[data.obs.group.=="A",:])
		end
		ref_obs_ind = findall(==("A"), data.obs.group)
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=1:3:P, obs_ind=ref_obs_ind)
		@test forward!(f_job) == forward!(ref_job)
		# NB: Projection should fail unless the var IDs are identical!


		f_job = Jobs.filter_var("name"=>>("D"), data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[data.var.name.>"D", :]
			@test isequal(f.var, filter("name"=>>("D"), data.var))
			test_dataframe_columns_identical("f.obs vs data.obs", f.obs, data.obs)
		end
		@test forward!(Jobs.get_obs(f_job)) == obs_spec_forwarded
		ref_var_ind = findall(>=("D"), data.var.name)
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=ref_var_ind)
		@test forward!(f_job) == forward!(ref_job)

		f_job = Jobs.filter_matrix("name"=>>("D"), 1:10:N, data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[data.var.name.>"D", 1:10:N]
			@test isequal(f.var, filter("name"=>>("D"), data.var))
			@test isequal(f.obs, data.obs[1:10:N,:])
		end
		ref_var_ind = findall(>=("D"), data.var.name)
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=ref_var_ind, obs_ind=1:10:N)
		@test forward!(f_job) == forward!(ref_job)
		# NB: Projection should fail unless the obs IDs are identical!

		f_job = Jobs.filter_matrix("name"=>>("D"), "group"=>==("A"), data_job)
		let f = fetch!(f_job)
			@test unblockify(materialize(f)) ≈ X[data.var.name.>"D", data.obs.group.=="A"]
			@test isequal(f.var, filter("name"=>>("D"), data.var))
			@test isequal(f.obs, filter("group"=>==("A"), data.obs))
		end
		ref_var_ind = findall(>=("D"), data.var.name)
		ref_obs_ind = findall(==("A"), data.obs.group)
		ref_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=ref_var_ind, obs_ind=ref_obs_ind)
		@test forward!(f_job) == forward!(ref_job)


		# annotations
		let var_mask = var_mask = in(var_annot_df.name).(data.var.name)
			f1_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; var_ind=findall(var_mask))
			f2_job = Jobs.filter_var(var_annot_df=>!ismissing, data_job)
			f3_job = Jobs.filter_var(var_annot_spec=>!ismissing, data_job)

			@test isequal(forward!(f1_job), forward!(f2_job))
			@test isequal(forward!(f1_job), forward!(f3_job))

			let f = fetch!(f3_job)
				@test unblockify(materialize(f)) ≈ X[var_mask, :]
				@test isequal(f.var, data.var[var_mask, :])
				test_dataframe_columns_identical("f.obs vs data.obs", f.obs, data.obs)
			end
		end
		let obs_mask = obs_mask = in(obs_annot_df.barcode).(data.obs.barcode)
			f1_job = SingleCellProjections.create_datamatrix_getindex_spec(data_job; obs_ind=findall(obs_mask))
			f2_job = Jobs.filter_obs(obs_annot_df=>!ismissing, data_job)
			f3_job = Jobs.filter_obs(obs_annot_spec=>!ismissing, data_job)

			@test isequal(forward!(f1_job), forward!(f2_job))
			@test isequal(forward!(f1_job), forward!(f3_job))

			let f = fetch!(f3_job)
				@test unblockify(materialize(f)) ≈ X[:, obs_mask]
				test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
				@test isequal(f.obs, data.obs[obs_mask, :])
			end
		end
	end
end
