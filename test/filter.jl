using Test
using SingleCellProjections
using SingleCellProjections: create_datamatrix_getindex_spec
using ReproducibleJobs: fetch!, forward!
using DataFrames
using Random: randperm

function run_filter_tests()
	@testset "Filtering" begin
		P,N = (50,587)

		counts_job = Jobs.load_counts(h5_path; sample_names="a")
		counts_job = Jobs.add_obs_column(counts_job, "group", counts_obs_group)
		counts_job = Jobs.add_obs_column(counts_job, "value", counts_obs_value)

		normalized_job = Jobs.normalize_matrix(Jobs.sctransform(counts_job))
		reduced_job = Jobs.pca(normalized_job; nsv=10, seed=1234)


		var_annot_df = select(fetch!(Jobs.get_var(counts_job)), ["id", "name"])[end:-4:1, :]
		var_annot_spec = SingleCellProjections.table_getindex_spec(Jobs.annotation(Jobs.get_var(counts_job), "name"), P:-4:1)

		obs_annot_df = select(fetch!(Jobs.get_obs(counts_job)), ["cell_id", "barcode"])[end:-5:1, :]
		obs_annot_spec = SingleCellProjections.table_getindex_spec(Jobs.annotation(Jobs.get_obs(counts_job), "barcode"), N:-5:1)


		# TODO: projections
		# TODO: test forwarding
		# TODO: test hash stability

		# TODO: Test for more data matrices
		@testset "filter $name" for (name,data_job) in (("counts",counts_job), ("normalized",normalized_job))
		# @testset "filter $name" for (name,data_job) in (("counts",counts_job),)
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

		@testset "getindex collapsing" begin
			f1 = "value"=>>(1.5)
			f2 = "group"=>==("A")

			# Filtering in different orders should forward! to the same spec and produce the same result
			j21 = Jobs.filter_obs(f1, Jobs.filter_obs(f2, counts_job))
			j12 = Jobs.filter_obs(f2, Jobs.filter_obs(f1, counts_job))

			@test isequal(forward!(Jobs.get_matrix(j21)), forward!(Jobs.get_matrix(j12))) # collapsing of matrix indexing
			@test isequal(forward!(Jobs.get_var(j21)), forward!(Jobs.get_var(j12))) # easier, vars are not filtered
			@test isequal(forward!(Jobs.get_obs(j21)), forward!(Jobs.get_obs(j12))) # collapsing of table indexing

			let r21 = fetch!(j21), r12 = fetch!(j12)
				# @show size(r21)
				@test isequal(r21.obs, r12.obs)
				@test unblockify(r21.matrix) == unblockify(r12.matrix)
			end

			f3 = "value"=><(1.8)
			j321 = Jobs.filter_obs(f1, Jobs.filter_obs(f2, Jobs.filter_obs(f3, counts_job)))
			j123 = Jobs.filter_obs(f3, Jobs.filter_obs(f2, Jobs.filter_obs(f1, counts_job)))

			@test isequal(forward!(Jobs.get_matrix(j321)), forward!(Jobs.get_matrix(j123))) # collapsing of matrix indexing
			@test isequal(forward!(Jobs.get_var(j321)), forward!(Jobs.get_var(j123))) # easier, vars are not filtered
			@test isequal(forward!(Jobs.get_obs(j321)), forward!(Jobs.get_obs(j123))) # collapsing of table indexing

			# @show forward!(Jobs.get_obs(j321))
			# @show forward!(Jobs.get_obs(j123))

			let r321 = fetch!(j321), r123 = fetch!(j123)
				# @show size(r321)
				@test isequal(r321.obs, r123.obs)
				@test unblockify(r321.matrix) == unblockify(r123.matrix)
			end
		end

		@testset "getindex Colon() collapsing n=$n" for n in 1:3
			job = counts_job
			for i in 1:n
				job = Jobs.filter_obs(:, job)
			end
			@test isequal(forward!(Jobs.get_matrix(job)), forward!(Jobs.get_matrix(counts_job)))
			@test isequal(forward!(Jobs.get_var(job)), forward!(Jobs.get_var(counts_job)))
			@test isequal(forward!(Jobs.get_obs(job)), forward!(Jobs.get_obs(counts_job)))
		end

		@testset "getindex no-op collapsing n=$n" for n in 1:3
			job = counts_job
			for i in 1:n
				job = Jobs.filter_obs(1:N, job)
			end
			@test isequal(forward!(Jobs.get_matrix(job)), forward!(Jobs.get_matrix(counts_job)))
			@test isequal(forward!(Jobs.get_var(job)), forward!(Jobs.get_var(counts_job)))
			# @test isequal(forward!(Jobs.get_obs(job)), forward!(Jobs.get_obs(counts_job))) # Do we want this to hold? Then we need to use simplify_ind for get_index/table_getindex.
		end

		@testset "getindex perm/invperm collapsing" begin
			rng = StableRNG(8080)
			perm = randperm(N)
			iperm = invperm(perm)

			jp = create_datamatrix_getindex_spec(counts_job; obs_ind=perm)
			job = create_datamatrix_getindex_spec(jp; obs_ind=iperm)
			@test fetch!(job) == fetch!(counts_job)

			# The matrix and obs cannot easily be collapsed. And it's an obscure case in practice. So we accept they do not collapse.
			@test isequal(forward!(Jobs.get_var(job)), forward!(Jobs.get_var(counts_job)))
		end
	end
end
