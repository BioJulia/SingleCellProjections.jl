using Test
using SingleCellProjections
using SingleCellProjections: SCPCore
import SingleCellProjections.Impl as Impl
using .SCPCore: Blocks
using ReproducibleJobs: fetch!, forward!
using DataFrames

function relative_std_ref(X; kwargs...)
	s = vec(std(X; kwargs...));
	s ./ maximum(s)
end

function run_sum_squared_tests()
	@testset "Sum Squared" begin
		counts_job = SCP.load_counts(h5_path; sample_names="a")
		normalized_job = SCP.normalize_matrix(SCP.sctransform(counts_job))

		counts_sub_job = SCP.load_counts(h5_subset_path; sample_names="p")
		normalized_sub_job = SCP.project(normalized_job, counts_job => counts_sub_job)

		# TODO: test forwarding
		# TODO: test hash stability

		@testset "variance and std $name" for (name, data_job) in (("normalized", normalized_job), ("counts", counts_job))
			data = fetch!(data_job)
			X = convert(Matrix{Float64}, unblockify(materialize(data)))
			P, N = size(data)
			id_col = only(names(data.var,1))

			mean_kwargs = name == "normalized" ? (;) : (mean=zeros(P,1),)

			@testset "$col" for (f, g, col) in (
					(SCP.variance,     var,              "variance"),
					(SCP.std,          std,              "std"),
					(SCP.relative_std, relative_std_ref, "relative_std"))
				expected = vec(g(X; dims=2, mean_kwargs...))
				job = f(data_job; assume_centered=true)
				result = fetch!(job)
				@test result isa DataFrame
				@test names(result) == [id_col, col]
				@test isequal(result[!, id_col], data.var[!, id_col])
				@test result[!, col] ≈ expected

				job2 = f(data_job; assume_centered=true, col="my_$col")
				result2 = fetch!(job2)
				@test names(result2) == [id_col, "my_$col"]
				@test result2[!, "my_$col"] ≈ expected

				@test_throws "assume_centered must be `true`" fetch!(f(data_job; assume_centered=false))
			end
		end

		@testset "projections" begin
			data_base = fetch!(normalized_job)
			data_proj = fetch!(normalized_sub_job)
			X_proj = convert(Matrix{Float64}, unblockify(materialize(data_proj)))
			id_col = only(names(data_base.var, 1))

			# NB: mean=0 because centering is already done using the mean from the base data
			@testset "$col projections" for (f, g, col) in (
					(SCP.variance,     var,              "variance"),
					(SCP.std,          std,              "std"),
					(SCP.relative_std, relative_std_ref, "relative_std"))
				# project=:no (default): uses base data regardless of projection
				base_job = f(normalized_job; assume_centered=true)
				proj_job = SCP.project(base_job, counts_job => counts_sub_job)
				@test isequal(forward!(base_job), forward!(proj_job))

				v = SCP.value_column_data(base_job)
				v_p = SCP.project(v)
				@test isequal(forward!(v), forward!(v_p))

				base = fetch!(base_job)
				proj = fetch!(proj_job)
				@test names(proj) == [id_col, col]
				@test proj[!, col] ≈ base[!, col]

				# project=:yes: uses projected data
				proj_yes_job = SCP.project(f(normalized_job; assume_centered=true, project=:yes), normalized_job => normalized_sub_job)
				result_yes = fetch!(proj_yes_job)
				@test names(result_yes) == [id_col, col]
				@test result_yes[!, col] ≈ vec(g(X_proj; mean=zeros(size(X_proj,1),1), dims=2))
			end
		end


		@testset "normalize_matrix annotate_variance" begin
			base = fetch!(normalized_job)
			X = convert(Matrix{Float64}, unblockify(materialize(base)))
			id_col = only(names(base.var, 1))

			@testset "$col" for (col_kwarg, g, col) in (
					(:variance_col,     var,              "variance"),
					(:std_col,          std,              "std"),
					(:relative_std_col, relative_std_ref, "relative_std"))
				expected = vec(g(X; dims=2))
				job = SCP.normalize_matrix(SCP.sctransform(counts_job); col_kwarg=>col)
				data = fetch!(job)
				@test data isa DataMatrix
				@test col in names(data.var)
				@test data.var[!, col] ≈ expected

				job2 = SCP.normalize_matrix(SCP.sctransform(counts_job); col_kwarg=>"my_$col")
				data2 = fetch!(job2)
				@test "my_$col" in names(data2.var)
				@test data2.var[!, "my_$col"] ≈ expected
			end

			@testset "annotate_* shortcut" begin
				job = SCP.normalize_matrix(SCP.sctransform(counts_job);
					annotate_variance=true, annotate_std=true, annotate_relative_std=true)
				data = fetch!(job)
				@test "variance" in names(data.var)
				@test "std" in names(data.var)
				@test "relative_std" in names(data.var)
				@test data.var[!, "variance"] ≈ fetch!(SCP.variance(normalized_job; assume_centered=true))[!, "variance"]
				@test data.var[!, "std"] ≈ fetch!(SCP.std(normalized_job; assume_centered=true))[!, "std"]
				@test data.var[!, "relative_std"] ≈ fetch!(SCP.relative_std(normalized_job; assume_centered=true))[!, "relative_std"]
			end

			@testset "variance annotation preserved under projection" begin
				job_with_var = SCP.normalize_matrix(SCP.sctransform(counts_job); annotate_variance=true, annotate_std=true, annotate_relative_std=true)
				proj_job = SCP.project(job_with_var, counts_job => counts_sub_job)
				data_base = fetch!(job_with_var)
				data_proj = fetch!(proj_job)
				@test "variance" in names(data_proj.var)
				@test isequal(data_proj.var[!, "variance"], data_base.var[!, "variance"])
				@test isequal(data_proj.var[!, "std"], data_base.var[!, "std"])
				@test isequal(data_proj.var[!, "relative_std"], data_base.var[!, "relative_std"])
			end

			@testset "center=false error" begin
				@test_throws "requires center=true" fetch!(SCP.normalize_matrix(SCP.sctransform(counts_job); center=false, annotate_variance=true))
			end
		end


		# col_sum_squared (per-obs sum of squares, dims=1, no mask) computed per block and vcat'd, so a
		# recurring sample reuses its cached block result. Internal helper (used by local_outlier_factor).
		@testset "col_sum_squared blocked" begin
			multi_job = SCP.load_counts([h5_path, mtx_path]; sample_names=["a","b"])
			data = fetch!(multi_job)
			@test data.matrix isa Blocks
			X = convert(Matrix{Float64}, unblockify(materialize(data)))
			mat = SCP.get_matrix(multi_job)

			# correctness on blocked input (holds before and after the per-block refactor)
			@test fetch!(Impl.col_sum_squared_blocked_job(mat)) ≈ vec(sum(abs2, X; dims=1))

			# per-block structure: a vcat over one (cached) col_sum_squared job per block
			fw = forward!(Impl.col_sum_squared_blocked_job(mat))
			@test fw.f === Impl.vcat_impl
			blocks = fw.args[1]
			@test length(blocks) == 2
			@test blocks[1].args[1].f === SCPCore.col_sum_squared   # inner job, under the `cached` wrapper
			@test !isequal(blocks[1], blocks[2])                    # distinct samples -> distinct jobs

			# cross-dataset dedup: same sample twice -> shared, deduplicated per-block job
			dup = SCP.get_matrix(SCP.load_counts([h5_path, h5_path]; sample_names=["a","b"]))
			bd = forward!(Impl.col_sum_squared_blocked_job(dup)).args[1]
			@test bd[1] === bd[2]
		end

		# row_sum_squared (per-var sum of squares, dims=2, no mask): each column block yields a partial
		# per-var sum of squares, combined element-wise with `sum`. Internal helper (used by variance/std).
		@testset "row_sum_squared blocked" begin
			multi_job = SCP.load_counts([h5_path, mtx_path]; sample_names=["a","b"])
			data = fetch!(multi_job)
			@test data.matrix isa Blocks
			X = convert(Matrix{Float64}, unblockify(materialize(data)))
			mat = SCP.get_matrix(multi_job)

			# correctness on blocked input (holds before and after the per-block refactor)
			@test fetch!(Impl.row_sum_squared_blocked_job(mat)) ≈ vec(sum(abs2, X; dims=2))

			# per-block structure: an element-wise `sum` over one (cached) row_sum_squared job per block
			fw = forward!(Impl.row_sum_squared_blocked_job(mat))
			@test fw.f === Impl.apply_impl && fw.args[1] === sum
			blocks = fw.args[2]
			@test length(blocks) == 2
			@test blocks[1].args[1].f === SCPCore.row_sum_squared   # inner job, under the `cached` wrapper
			@test !isequal(blocks[1], blocks[2])                    # distinct samples -> distinct jobs

			# cross-dataset dedup: same sample twice -> shared, deduplicated per-block job
			dup = SCP.get_matrix(SCP.load_counts([h5_path, h5_path]; sample_names=["a","b"]))
			bd = forward!(Impl.row_sum_squared_blocked_job(dup)).args[2]
			@test bd[1] === bd[2]
		end

	end
end
