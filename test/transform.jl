@testset "Transforms" begin
	P,N = (50,587)

	# TODO: Test .mtx file (implement with specs first!)
	counts_job = Jobs.load_counts(h5_path; sample_names="a")


	@testset "logtransform scale_factor=$scale_factor T=$T" for scale_factor in (10_000, 1_000), T in (Float64,Float32)
		X = simple_logtransform(expected_mat, scale_factor)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)
		if T==Float64
			l_job = Jobs.logtransform(counts_job; kwargs...)
		else
			l_job = Jobs.logtransform(T, counts_job; kwargs...)
			X = T.(X)
		end

		let l = fetch!(l_job), counts = fetch!(counts_job)
			@test l.matrix.matrix ≈ X
			@test nnz(l.matrix.matrix) == expected_nnz
			@test eltype(l.matrix.matrix) == T

			@test isequal(l.var, counts.var)

			# lproj = project(counts_proj, l)
			# @test lproj.matrix.matrix ≈ X[:,proj_obs_indices]
			# @test eltype(lproj.matrix.matrix) == T

			# test_show(l; matrix="SparseMatrixCSC", var=names(counts.var), obs=names(counts.obs), models="LogTransformModel")
			# test_show(lproj; matrix="SparseMatrixCSC", var=names(counts_proj.var), obs=names(counts_proj.obs), models="LogTransformModel")

			# Variable subsetting
			@testset "var_filter" begin
				var_mask = counts.var.name .> "L"
				X_f = T.(simple_logtransform(expected_mat[var_mask,:], scale_factor))

				l_job_f = Jobs.logtransform(T, counts_job; var_filter="name"=>>("L"), kwargs...)
				let l_f = fetch!(l_job_f)
					@test l_f.matrix.matrix ≈ X_f
					@test eltype(l_f.matrix.matrix) == T

					@test isequal(l_f.var, counts.var[var_mask,:])
				end
			end
		end
	end

end
