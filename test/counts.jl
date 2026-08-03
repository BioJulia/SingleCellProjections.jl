using Test
using SingleCellProjections
import SingleCellProjections as SCP
using SingleCellProjections: SCPCore
import SingleCellProjections.Impl as Impl
using .SCPCore: unblockify, Blocks
using ReproducibleJobs: fetch!, forward!, SpecRef, get_cached
using DataFrames


# Navigate a forwarded DataMatrix spec to the job that computes obs column `col`.
# The forwarded spec is `DataMatrix(matrix, var_table, obs_table)`, where `obs_table` is a
# `create_table(col => value, ...)` call. Assert that shape before indexing into it.
function obs_value_spec(fw, col)
	@assert fw isa SpecRef && fw.f === SCPCore.DataMatrix "expected a forwarded DataMatrix spec, got $fw"
	obs_tbl = fw.args[end]          # the obs `create_table` call
	@assert obs_tbl isa SpecRef && obs_tbl.f === Impl.create_table "expected obs create_table, got $obs_tbl"
	for a in obs_tbl.args
		a isa Pair && a.first == col && return a.second
	end
	error("obs column \"$col\" not found in forwarded spec")
end

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
	end
end
