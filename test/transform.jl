@testset "Transforms" begin
	P,N = (50,587)

	# TODO: Test .mtx file (implement with specs first!)
	counts_job = Jobs.load_counts(h5_path; sample_names="a")

	# TODO: projections
	# TODO: test forwarding
	# TODO: test hash stability

	@testset "logtransform scale_factor=$scale_factor T=$T" for scale_factor in (10_000, 1_000), T in (Float64,Float32)
		X = simple_logtransform(expected_mat, scale_factor)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)
		if T==Float64
			l_job = Jobs.logtransform(counts_job; kwargs...)
		else
			l_job = Jobs.logtransform(T, counts_job; kwargs...)
			X = T.(X)
		end

		@test forward(Jobs.get_obs(l_job)).spec == forward(Jobs.get_obs(counts_job)).spec
		@test forward(Jobs.get_var(l_job)).spec == forward(Jobs.get_var(counts_job)).spec

		let l = fetch!(l_job), counts = fetch!(counts_job)
			@test l.matrix.matrix ≈ X
			@test nnz(l.matrix.matrix) == expected_nnz
			@test eltype(l.matrix.matrix) == T

			@test isequal(l.var, counts.var)
			@test isequal(l.obs, counts.obs)

			@test l.obs.cell_id === counts.obs.cell_id
			@test l.var.id === counts.var.id

			# Variable subsetting
			@testset "var_filter" begin
				var_mask = counts.var.name .> "L"
				X_filtered = T.(simple_logtransform(expected_mat[var_mask,:], scale_factor))

				l_filtered_job = Jobs.logtransform(T, counts_job; var_filter="name"=>>("L"), kwargs...)

				@test forward(Jobs.get_obs(l_filtered_job)).spec == forward(Jobs.get_obs(counts_job)).spec

				let l_filtered = fetch!(l_filtered_job)
					@test l_filtered.matrix.matrix ≈ X_filtered
					@test eltype(l_filtered.matrix.matrix) == T

					@test isequal(l_filtered.var, counts.var[var_mask,:])
					@test isequal(l_filtered.obs, counts.obs)
				end
			end
		end
	end

end
