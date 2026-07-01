using Test
using SingleCellProjections
import SingleCellProjections as SCP
using SingleCellProjections: SCPCore
using .SCPCore: unblockify
using ReproducibleJobs: fetch!
using DataFrames
using SparseArrays


# Independent, dependency-free tied-rank Mann-Whitney U for group 1 (matches the U reported by the kernel).
function _tiedrank(v)
	p = sortperm(v)
	r = zeros(Float64, length(v))
	i = 1
	while i <= length(v)
		j = i
		while j < length(v) && isequal(v[p[j+1]], v[p[i]]); j += 1; end
		avg = (i+j)/2
		for k in i:j; r[p[k]] = avg; end
		i = j+1
	end
	r
end
function _u_ref(x, mask1, mask2)
	r = _tiedrank(vcat(x[mask1], x[mask2]))
	n1 = count(mask1)
	sum(r[1:n1]) - n1*(n1+1)/2
end


function run_mannwhitney_tests()
	@testset "Mann-Whitney U-test" begin
		counts_job = SCP.load_counts(h5_path; sample_names="a")
		counts_job = SCP.add_obs_column(counts_job, "group", counts_obs_group)
		# group2 has exactly two non-missing values (A, B) - used to exercise auto-detection
		counts_job = SCP.add_obs_column(counts_job, "group2", replace(counts_obs_group, "C"=>missing))
		l_job = SCP.logtransform(counts_job)

		l = fetch!(l_job)
		X = convert(Matrix{Float64}, unblockify(materialize(l)))
		idcol = only(names(l.var, 1))

		# reference U/p via the (trusted) kernel, but with an INDEPENDENTLY constructed groups vector
		ref(groups) = SCPCore.mannwhitney_sparse(sparse(X), groups)

		@testset "$desc" for (desc, args, groups) in (
				("A vs B",       ("group", "A", "B"), [g=="A" ? 1 : g=="B" ? 2 : 0 for g in l.obs.group]),
				("A vs rest",    ("group", "A"),      [g=="A" ? 1 : 2 for g in l.obs.group]),
				("auto-detect",  ("group2",),         [isequal(g,"A") ? 1 : isequal(g,"B") ? 2 : 0 for g in l.obs.group2]),
			)
			r = fetch!(SCP.mannwhitney(l_job, args...))
			U, p = ref(groups)

			@test r isa DataFrame
			@test names(r) == [idcol, "U", "pValue"]
			@test isequal(r[!, idcol], l.var[!, idcol])
			@test r.U ≈ U
			@test r.pValue ≈ p

			# spot-check U against a fully independent tied-rank computation
			ind = 7
			@test r.U[ind] ≈ _u_ref(X[ind, :], groups .== 1, groups .== 2)
		end

		@testset "column names" begin
			r = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"; statistic_col="my_u", pvalue_col="my_p"))
			@test names(r) == [idcol, "my_u", "my_p"]

			r = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"; pvalue_col=nothing))
			@test names(r) == [idcol, "U"]

			r = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"; statistic_col=nothing))
			@test names(r) == [idcol, "pValue"]
		end

		@testset "missing handling" begin
			# :skip (default) excludes missing; :error throws
			r_skip = fetch!(SCP.mannwhitney(l_job, "group2", "A", "B"))
			groups = [isequal(g,"A") ? 1 : isequal(g,"B") ? 2 : 0 for g in l.obs.group2]
			U, p = ref(groups)
			@test r_skip.U ≈ U
			@test r_skip.pValue ≈ p
			@test_throws Exception fetch!(SCP.mannwhitney(l_job, "group2", "A", "B"; h1_missing=:error))
		end

		@testset "projection" begin
			counts_sub = SCP.load_counts(h5_subset_path; sample_names="a")
			counts_sub = SCP.add_obs_column(counts_sub, "group", counts_obs_group[pbmc_subset_ind])
			l_sub_job = SCP.logtransform(counts_sub)

			# NB: the 272-cell subset contains groups A and C (no B)
			mw_job = SCP.mannwhitney(l_job, "group", "A", "C")
			proj = fetch!(SCP.project(mw_job, l_job => l_sub_job))
			direct = fetch!(SCP.mannwhitney(l_sub_job, "group", "A", "C"))

			@test isequal(proj[!, idcol], direct[!, idcol])
			@test proj.U ≈ direct.U
			@test proj.pValue ≈ direct.pValue
		end
	end
end
