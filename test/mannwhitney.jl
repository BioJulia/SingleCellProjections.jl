using Test
using SingleCellProjections
import SingleCellProjections as SCP
using SingleCellProjections: SCPCore
using .SCPCore: unblockify
using ReproducibleJobs: fetch!
using DataFrames
import HypothesisTests


# Independent reference: per-variable Mann-Whitney U/z/p from HypothesisTests. `m.U` matches the
# kernel's group-1 U and `m.sigma` its tie-corrected σ, so the signed z-score is
# `(m.U - n1*n2/2)/m.sigma` (guarding the fully-tied σ==0 case to match the kernel).
function _mw_ref(X, groups)
	n1 = count(==(1), groups)
	n2 = count(==(2), groups)
	mw = [HypothesisTests.ApproximateMannWhitneyUTest(X[i, groups.==1], X[i, groups.==2]) for i in axes(X,1)]
	U = getfield.(mw, :U)
	z = [m.sigma > 0 ? (m.U - n1*n2/2)/m.sigma : 0.0 for m in mw]
	p = HypothesisTests.pvalue.(mw)
	U, z, p
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

		@testset "$desc" for (desc, args, groups) in (
				("A vs B",       ("group", "A", "B"), [g=="A" ? 1 : g=="B" ? 2 : 0 for g in l.obs.group]),
				("A vs rest",    ("group", "A"),      [g=="A" ? 1 : 2 for g in l.obs.group]),
				("auto-detect",  ("group2",),         [isequal(g,"A") ? 1 : isequal(g,"B") ? 2 : 0 for g in l.obs.group2]),
			)
			U, z, p = _mw_ref(X, groups)

			# do_sort=false keeps original variable order so it aligns with the reference
			r = fetch!(SCP.mannwhitney(l_job, args...; do_sort=false, z_col="z"))

			@test r isa DataFrame
			@test names(r) == [idcol, "U", "z", "pValue"]
			@test isequal(r[!, idcol], l.var[!, idcol])
			@test r.U ≈ U
			@test r.z ≈ z
			@test r.pValue ≈ p
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
			r_skip = fetch!(SCP.mannwhitney(l_job, "group2", "A", "B"; do_sort=false))
			groups = [isequal(g,"A") ? 1 : isequal(g,"B") ? 2 : 0 for g in l.obs.group2]
			U, _, p = _mw_ref(X, groups)
			@test r_skip.U ≈ U
			@test r_skip.pValue ≈ p
			@test_throws Exception fetch!(SCP.mannwhitney(l_job, "group2", "A", "B"; h1_missing=:error))
		end

		@testset "var_cols" begin
			# extra var columns are carried through, inserted before the statistic columns
			r = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"; var_cols="name", do_sort=false))
			@test names(r) == [idcol, "name", "U", "pValue"]
			@test isequal(r.name, l.var.name)

			r = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"; var_cols=("name","feature_type"), do_sort=false))
			@test names(r) == [idcol, "name", "feature_type", "U", "pValue"]
		end

		@testset "z-score sorting and z_col" begin
			groups = [g=="A" ? 1 : g=="B" ? 2 : 0 for g in l.obs.group]
			U, z, p = _mw_ref(X, groups)
			perm = sortperm(abs.(z); rev=true) # |z| descending == significance order

			# z_col adds a signed z column; results are sorted by |z| (most significant first)
			r = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"; z_col="z"))
			@test names(r) == [idcol, "U", "z", "pValue"]
			@test issorted(abs.(r.z); rev=true)
			@test isequal(r[!, idcol], l.var[perm, idcol])
			@test r.z ≈ z[perm]
			@test r.U ≈ U[perm]
			@test r.pValue ≈ p[perm]

			# z is omitted by default, but the |z| sort still happens
			r_default = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"))
			@test names(r_default) == [idcol, "U", "pValue"]
			@test isequal(r_default[!, idcol], r[!, idcol])

			# do_sort=false keeps the original variable order
			r_unsorted = fetch!(SCP.mannwhitney(l_job, "group", "A", "B"; do_sort=false))
			@test isequal(r_unsorted[!, idcol], l.var[!, idcol])
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
