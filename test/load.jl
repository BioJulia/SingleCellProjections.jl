using Test
using SingleCellProjections
using ReproducibleJobs: fetch!, forward!
using SparseArrays

function run_load_tests()
	@testset "load_counts" begin
		P,N = (50,587)

		counts_job = SCP.load_counts(h5_path; sample_names="a")

		counts_sub_job = SCP.load_counts(h5_subset_path; sample_names="p")


		# Test result
		let counts = fetch!(counts_job)
			@test size(counts)==(P,N)
			@test nnz(unblockify(counts.matrix)) == expected_nnz

			@test names(counts.obs) == ["cell_id", "sample_name", "barcode"]
			@test counts.obs.cell_id == string.("a_",expected_barcodes)
			@test counts.obs.sample_name == fill("a",N)
			@test counts.obs.barcode == expected_barcodes

			@test names(counts.var) == ["id", "name", "feature_type", "genome"]
			@test counts.var.id == expected_feature_ids
			@test counts.var.name == expected_feature_names
			@test counts.var.feature_type == expected_feature_types
			@test counts.var.genome == expected_feature_genome

			@test unblockify(counts.matrix) == expected_mat
			@test unblockify(counts.matrix) isa SparseMatrixCSC{Int64,Int32}
		end

		@testset "Projection top-level replacements" begin
			p_job = SCP.project(counts_job, counts_job=>counts_sub_job)
			@test isequal(forward!(p_job), forward!(counts_sub_job))

			matrix_job = SCP.get_matrix(counts_job)
			matrix_sub_job = SCP.get_matrix(counts_sub_job)
			p_matrix_job = SCP.project(matrix_job, matrix_job=>matrix_sub_job)
			@test isequal(forward!(p_matrix_job), forward!(matrix_sub_job))

			var_job = SCP.get_var(counts_job)
			var_sub_job = SCP.get_var(counts_sub_job)
			p_var_job = SCP.project(var_job, var_job=>var_sub_job)
			@test isequal(forward!(p_var_job), forward!(var_sub_job))

			obs_job = SCP.get_obs(counts_job)
			obs_sub_job = SCP.get_obs(counts_sub_job)
			p_obs_job = SCP.project(obs_job, obs_job=>obs_sub_job)
			@test isequal(forward!(p_obs_job), forward!(obs_sub_job))
		end

		@testset "Matrix Market (.mtx)" begin
			mtx_job = SCP.load_counts(mtx_path; sample_names="a")
			let mtx = fetch!(mtx_job)
				@test size(mtx) == (P,N)
				@test unblockify(mtx.matrix) == expected_mat
				@test unblockify(mtx.matrix) isa SparseMatrixCSC{Int64,Int32}

				# obs is identical to the .h5 load
				@test names(mtx.obs) == ["cell_id", "sample_name", "barcode"]
				@test mtx.obs.cell_id == string.("a_", expected_barcodes)
				@test mtx.obs.sample_name == fill("a",N)
				@test mtx.obs.barcode == expected_barcodes

				# var: features.tsv has no `genome` column, but the shared columns match the .h5 load
				@test names(mtx.var) == ["id", "name", "feature_type"]
				@test mtx.var.id == expected_feature_ids
				@test mtx.var.name == expected_feature_names
				@test mtx.var.feature_type == expected_feature_types
			end

			feat_path = joinpath(dirname(mtx_path), "features.tsv.gz")
			bc_path   = joinpath(dirname(mtx_path), "barcodes.tsv.gz")

			# Specifying paths identical to the guessed paths should result in identical jobs (after forwarding)
			mtx_job2 = SCP.load_counts(mtx_path; sample_names="a", feature_filenames=feat_path, barcode_filenames=bc_path)
			@test forward!(mtx_job2) === forward!(mtx_job)

			# a mismatched number of explicit filenames is an error
			@test_throws ArgumentError SCP.load_counts([mtx_path]; sample_names="a", feature_filenames=[feat_path, feat_path])
			@test_throws ArgumentError SCP.load_counts([mtx_path]; sample_names="a", barcode_filenames=[bc_path, bc_path])
		end

		@testset "Mixed .h5 and .mtx" begin
			# load one .h5 sample and one .mtx sample in the same call (same underlying data)
			mixed = fetch!(SCP.load_counts([h5_path, mtx_path]; sample_names=["a","b"]))
			@test size(mixed) == (P, 2N)

			# obs: both samples merged, in order
			@test names(mixed.obs) == ["cell_id", "sample_name", "barcode"]
			@test mixed.obs.sample_name == [fill("a",N); fill("b",N)]
			@test mixed.obs.cell_id == [string.("a_",expected_barcodes); string.("b_",expected_barcodes)]

			# var: `id`/`name`/`feature_type` agree across both samples and are kept. `genome` exists
			# only in the .h5 features, so per gene the value is inconsistent across the two samples and
			# `combine_var` marks it "ambiguous" (its sentinel for annotations that differ between samples).
			@test names(mixed.var) == ["id", "name", "feature_type", "genome"]
			@test mixed.var.id == expected_feature_ids
			@test mixed.var.name == expected_feature_names
			@test mixed.var.feature_type == expected_feature_types
			@test all(==("ambiguous"), mixed.var.genome)

			# each sample's block equals the expected matrix
			M = unblockify(mixed.matrix)
			@test M[:, mixed.obs.sample_name .== "a"] == expected_mat
			@test M[:, mixed.obs.sample_name .== "b"] == expected_mat
		end
	end
end
