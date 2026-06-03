using Test
using SingleCellProjections
using ReproducibleJobs: fetch!, forward!
using SparseArrays

function run_load_tests()
	@testset "load_counts" begin
		P,N = (50,587)

		# TODO: Test .mtx file (implement with specs first!)
		counts_job = Jobs.load_counts(h5_path; sample_names="a")

		counts_sub_job = Jobs.load_counts(h5_subset_path; sample_names="p")


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
			p_job = Jobs.project(counts_job, counts_job=>counts_sub_job)
			@test isequal(forward!(p_job), forward!(counts_sub_job))

			matrix_job = Jobs.get_matrix(counts_job)
			matrix_sub_job = Jobs.get_matrix(counts_sub_job)
			p_matrix_job = Jobs.project(matrix_job, matrix_job=>matrix_sub_job)
			@test isequal(forward!(p_matrix_job), forward!(matrix_sub_job))

			var_job = Jobs.get_var(counts_job)
			var_sub_job = Jobs.get_var(counts_sub_job)
			p_var_job = Jobs.project(var_job, var_job=>var_sub_job)
			@test isequal(forward!(p_var_job), forward!(var_sub_job))

			obs_job = Jobs.get_obs(counts_job)
			obs_sub_job = Jobs.get_obs(counts_sub_job)
			p_obs_job = Jobs.project(obs_job, obs_job=>obs_sub_job)
			@test isequal(forward!(p_obs_job), forward!(obs_sub_job))
		end
	end
end
