using Test
using SingleCellProjections
import SingleCellProjections as SCP
using SingleCellProjections: SCPCore
import SingleCellProjections.Impl as Impl
using .SCPCore: unblockify, Blocks
using ReproducibleJobs: fetch!, forward!, SpecRef, get_cached
using DataFrames


# Navigate a forwarded DataMatrix spec to the job that computes column `col` of the var or obs table.
# The forwarded spec is `DataMatrix(matrix, var_table, obs_table)`, where each table is a
# `create_table(col => value, ...)` call. Assert that shape before indexing into it.
function table_value_spec(fw, which, col)
	@assert fw isa SpecRef && fw.f === SCPCore.DataMatrix "expected a forwarded DataMatrix spec, got $fw"
	tbl = which === :obs ? fw.args[end] : fw.args[end-1]   # DataMatrix(matrix, var_table, obs_table)
	@assert tbl isa SpecRef && tbl.f === Impl.create_table "expected $which create_table, got $tbl"
	for a in tbl.args
		a isa Pair && a.first == col && return a.second
	end
	error("$which column \"$col\" not found in forwarded spec")
end
obs_value_spec(fw, col) = table_value_spec(fw, :obs, col)
var_value_spec(fw, col) = table_value_spec(fw, :var, col)

# Unwrap a `cached(...)` wrapper (`cached(job)` == `create_job(get_cached, job)`).
unwrap_cached(s) = (s isa SpecRef && s.f === get_cached) ? s.args[1] : s


function run_counts_tests()
	@testset "var/obs counts" begin
		# A genuinely block-structured matrix: two samples (same underlying data) loaded together
		# give a `Blocks` matrix with one column block per sample. The single-sample load is the
		# unblocked reference.
		multi_job = SCP.load_counts([h5_path, mtx_path]; sample_names=["a","b"])
		single_job = SCP.load_counts(h5_path; sample_names="a")

		data = fetch!(multi_job)
		@test data.matrix isa Blocks              # guard: the input really is blocked - RH, not enough, a single sample can get blocked as well, but this happens later. Testing at the spec level is what matters here.
		nblocks = size(data.matrix.blocks, 2)
		@test nblocks == 2 # same
		X = convert(Matrix{Float64}, unblockify(materialize(data)))  # P×2N dense reference

		@testset "var_counts_sum on blocked input" begin
			# all variables, identity
			r = fetch!(SCP.var_counts_sum(multi_job, "total"))
			@test r isa DataMatrix
			@test "total" in names(r.obs)
			@test r.obs.total ≈ vec(sum(X; dims=1))

			# nonzero-count via f=!iszero
			r2 = fetch!(SCP.var_counts_sum(!iszero, multi_job, "nnz"))
			@test r2.obs.nnz ≈ vec(sum(!iszero, X; dims=1))

			# variable subset (the row mask is identical for every column block)
			sub = Set(data.var.name[2:2:20])
			mask = in(sub).(data.var.name)
			r3 = fetch!(SCP.var_counts_sum(multi_job, "sub", "name"=>in(sub)))
			@test r3.obs.sub ≈ vec(sum(X[mask, :]; dims=1))
		end

		@testset "blocked ≡ unblocked" begin
			# The per-cell values for sample "a" must match computing on that sample alone.
			rm = fetch!(SCP.var_counts_sum(multi_job, "total"))
			rs = fetch!(SCP.var_counts_sum(single_job, "total"))
			@test rm.obs.total[rm.obs.sample_name .== "a"] ≈ rs.obs.total
		end

		@testset "reduction is mapped over blocks" begin
			# In the forwarded spec, the "total" column is a `vcat` over per-block `counts_sum` (one
			# job per block), rather than a single `counts_sum` over the whole `hblock`.
			fw = forward!(SCP.var_counts_sum(multi_job, "total"))
			v = unwrap_cached(obs_value_spec(fw, "total"))
			@test v isa SpecRef && v.f === Impl.vcat_impl
			@test !isequal(v.args[1][1], v.args[1][2])   # distinct samples -> distinct per-block jobs
		end

		@testset "shared sample block is deduplicated (cache reuse)" begin
			# The point of block support: loading the same sample twice yields identical block specs,
			# so the per-block reduction is a single (deduplicated) job reused for both blocks - i.e. a
			# recurring sample reuses its cached block result instead of recomputing.
			dup_job = SCP.load_counts([h5_path, h5_path]; sample_names=["a","b"])
			v = unwrap_cached(obs_value_spec(forward!(SCP.var_counts_sum(dup_job, "total")), "total"))
			@test v.f === Impl.vcat_impl
			blocks = v.args[1]                     # per-block reduction jobs
			@test length(blocks) == 2
			@test blocks[1] === blocks[2]          # repeated sample -> shared, deduplicated job
		end


		# --- obs_counts_sum: dims=2 (per-var result, reduced over obs) --------------------------------
		# Here the obs selection `ind` is over columns, so it must be split per block (each block owns a
		# disjoint set of obs) and the per-block partial per-var sums combined with an element-wise sum.
		@testset "obs_counts_sum on blocked input" begin
			# all obs, identity
			r = fetch!(SCP.obs_counts_sum(multi_job, "total"))
			@test r isa DataMatrix
			@test "total" in names(r.var)
			@test r.var.total ≈ vec(sum(X; dims=2))

			# nonzero-count via f=!iszero
			r2 = fetch!(SCP.obs_counts_sum(!iszero, multi_job, "nnz"))
			@test r2.var.nnz ≈ vec(sum(!iszero, X; dims=2))

			# obs subset: every other cell, so the mask genuinely splits *within* both blocks
			# (exercises per-block `ind` slicing).
			sub = Set(data.obs.cell_id[2:2:end])
			obsmask = in(sub).(data.obs.cell_id)
			@test count(obsmask[data.obs.sample_name .== "a"]) > 0   # guard: selects within both blocks
			@test count(obsmask[data.obs.sample_name .== "b"]) > 0
			r3 = fetch!(SCP.obs_counts_sum(multi_job, "sub", "cell_id"=>in(sub)))
			@test r3.var.sub ≈ vec(sum(X[:, obsmask]; dims=2))
		end

		@testset "obs_counts_sum blocked ≡ unblocked" begin
			# Summing over sample "a"'s obs in the blocked dataset must match computing on sample "a"
			# alone (block a of multi_job is the same underlying data as single_job).
			rm = fetch!(SCP.obs_counts_sum(multi_job, "sub_a", "sample_name"=>isequal("a")))
			rs = fetch!(SCP.obs_counts_sum(single_job, "total"))
			@test rm.var.sub_a ≈ rs.var.total
		end

		@testset "obs_counts_sum reduction is mapped over blocks" begin
			# The var "total" column is an element-wise `vsum` over per-block `counts_sum(dims=2)` jobs
			# (one partial per-var sum per block), not a single `counts_sum` over the whole `hblock`.
			fw = forward!(SCP.obs_counts_sum(multi_job, "total"))
			v = unwrap_cached(var_value_spec(fw, "total"))
			@test v isa SpecRef && v.f === Impl.vsum_impl
			@test !isequal(v.args[1][1], v.args[1][2])   # distinct samples -> distinct per-block jobs
		end

		@testset "obs_counts_sum shared sample block is deduplicated (cache reuse)" begin
			dup_job = SCP.load_counts([h5_path, h5_path]; sample_names=["a","b"])
			v = unwrap_cached(var_value_spec(forward!(SCP.obs_counts_sum(dup_job, "total")), "total"))
			@test v.f === Impl.vsum_impl
			blocks = v.args[1]                     # per-block reduction jobs
			@test length(blocks) == 2
			@test blocks[1] === blocks[2]          # repeated sample -> shared, deduplicated job
		end
	end
end
