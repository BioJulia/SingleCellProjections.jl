@testset "Filtering" begin
	counts_job = Jobs.load_counts(h5_path; sample_names="a")
	counts_job = Jobs.add_obs_column(counts_job, "group", counts_obs_group)
	counts_job = Jobs.add_obs_column(counts_job, "value", counts_obs_value)

	# TODO: projections
	# TODO: test forwarding
	# TODO: test hash stability

	# TODO: Test for more data matrices
	# @testset "filter $name" for (name,data_job) in (("counts",counts_job), ("normalized",normalized_job), ("reduced",reduced_job))
	@testset "filter $name" for (name,data_job) in (("counts",counts_job),)
		data = fetch!(data_job)
		data_spec_forwarded = forward(data_job).spec
		var_spec_forwarded = forward(Jobs.get_var(data_job)).spec
		obs_spec_forwarded = forward(Jobs.get_obs(data_job)).spec

		X = materialize(data)
		P,N = size(data)

		f_job = Jobs.filter_matrix(:, :, data_job)
		@test forward(f_job).spec == data_spec_forwarded
		let f = fetch!(f_job)
			@test materialize(f) ≈ X
			test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
			test_dataframe_columns_identical("f.obs vs data.obs", f.obs, data.obs)
		end

		f_job = Jobs.filter_var(1:2:P, data_job)
		@test forward(Jobs.get_obs(f_job)).spec == obs_spec_forwarded
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[1:2:P, :]
			@test isequal(f.var, data.var[1:2:P,:])
			test_dataframe_columns_identical("f.obs vs data.obs", f.obs, data.obs)
		end
		# NB: Projection should fail unless the var IDs are identical!

		f_job = Jobs.filter_obs(1:2:N, data_job)
		@test forward(Jobs.get_var(f_job)).spec == var_spec_forwarded
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[:, 1:2:N]
			test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
			@test isequal(f.obs, data.obs[1:2:N,:])
		end
		# NB: Projection should fail unless the obs IDs are identical!

		f_job = Jobs.filter_matrix(1:3:P, 1:10:N, data_job)
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[1:3:P, 1:10:N]
			@test isequal(f.var, data.var[1:3:P,:])
			@test isequal(f.obs, data.obs[1:10:N,:])
		end
		# NB: Projection should fail unless the var and obs IDs are identical!

		f_job = Jobs.filter_obs("group"=>==("A"), data_job)
		@test forward(Jobs.get_var(f_job)).spec == var_spec_forwarded
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[:, data.obs.group.=="A"]
			test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
			@test isequal(f.obs, filter("group"=>==("A"), data.obs))
		end

		# Hmm. How do we support this? The problem is the anonymous function that cannot be hashed currently. Implement HashableFunctions?
		# f_job = Jobs.filter_obs(["group","value"]=>(g,v)->g=="A" && v<1.0, data_job)
		# let f = fetch!(f_job)
		# 	@test materialize(f) ≈ X[:, (data.obs.group.=="A") .& (data.obs.value.<1.0)]
		# 	test_dataframe_columns_identical("f.var vs data.var", f.var, data.var)
		# 	@test isequal(f.obs, filter(["group","value"]=>(g,v)->g=="A" && v<1.0, data.obs))
		# end


		f_job = Jobs.filter_matrix(1:3:P, "group"=>==("A"), data_job)
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[1:3:P, data.obs.group.=="A"]
			@test isequal(f.var, data.var[1:3:P,:])
			@test isequal(f.obs, data.obs[data.obs.group.=="A",:])
		end
		# NB: Projection should fail unless the var IDs are identical!


		f_job = Jobs.filter_var("name"=>>("D"), data_job)
		@test forward(Jobs.get_obs(f_job)).spec == obs_spec_forwarded
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[data.var.name.>"D", :]
			@test isequal(f.var, filter("name"=>>("D"), data.var))
			test_dataframe_columns_identical("f.obs vs data.obs", f.obs, data.obs)
		end

		f_job = Jobs.filter_matrix("name"=>>("D"), 1:10:N, data_job)
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[data.var.name.>"D", 1:10:N]
			@test isequal(f.var, filter("name"=>>("D"), data.var))
			@test isequal(f.obs, data.obs[1:10:N,:])
		end
		# NB: Projection should fail unless the obs IDs are identical!

		f_job = Jobs.filter_matrix("name"=>>("D"), "group"=>==("A"), data_job)
		let f = fetch!(f_job)
			@test materialize(f) ≈ X[data.var.name.>"D", data.obs.group.=="A"]
			@test isequal(f.var, filter("name"=>>("D"), data.var))
			@test isequal(f.obs, filter("group"=>==("A"), data.obs))
		end


	end
	
end
