using Test
using SingleCellProjections
import SingleCellProjections as SCP
using SingleCellProjections: SCPCore
using .SCPCore: unblockify
using ReproducibleJobs: fetch!
using DataFrames


function run_ttest_tests()
	@testset "t-test" begin
		counts_job = SCP.load_counts(h5_path; sample_names="a")
		counts_job = SCP.add_obs_column(counts_job, "group", counts_obs_group)
		counts_job = SCP.add_obs_column(counts_job, "value", counts_obs_value)
		counts_job = SCP.add_obs_column(counts_job, "value2", counts_obs_value.^2)
		value3 = missings(Float64, length(counts_obs_value))
		value3[1:2:end] .= 1:cld(length(counts_obs_value),2)
		counts_job = SCP.add_obs_column(counts_job, "value3", value3)
		# two-level group (A/B) for the two-group t-test
		counts_job = SCP.add_obs_column(counts_job, "twogroup", replace(counts_obs_group, "C"=>"A"))
		# group2 has two non-missing values (A, B) - exercises categorical/twogroup missing handling
		counts_job = SCP.add_obs_column(counts_job, "group2", replace(counts_obs_group, "C"=>missing))
		l_job = SCP.logtransform(counts_job)

		l = fetch!(l_job)
		X = convert(Matrix{Float64}, unblockify(materialize(l)))
		obs = l.obs
		idcol = only(names(l.var, 1))
		nzmask = vec(any(!iszero, X; dims=2)) # exclude all-zero variables (t/p are NaN there)

		@testset "numerical H1:$h1 H0:$h0" for (h1, h0) in (
				("value",  ()),
				("value",  ("group",)),
				("value2", ("value",)),
				("value2", ("group","value")),
				("value",  ("value",)), # h1 ⊆ h0 -> t=0, p=1
			)
			gtT, gtP, gtβ = ttest_ground_truth(X, obs, h1, h0)
			r = fetch!(SCP.ttest(l_job, h1; h0, do_sort=false))
			@test names(r) == [idcol, "t", "pValue", "difference"]
			@test r.t[nzmask] ≈ gtT[nzmask]
			@test r.pValue[nzmask] ≈ gtP[nzmask]
			@test r.difference[nzmask] ≈ gtβ[nzmask]
		end

		@testset "twogroup $desc" for (desc, ga, gb) in (("A vs B","A","B"), ("B vs A","B","A"))
			gtT, gtP, gtβ = ttest_ground_truth(X, obs, "twogroup", ga, gb, ())
			r = fetch!(SCP.ttest(l_job, "twogroup"=>SCP.twogroup_covariate(ga,gb); do_sort=false))
			@test names(r) == [idcol, "t", "pValue", "difference"]
			@test r.t[nzmask] ≈ gtT[nzmask]
			@test r.pValue[nzmask] ≈ gtP[nzmask]
			@test r.difference[nzmask] ≈ gtβ[nzmask]
		end

		@testset "missing handling" begin
			# :error throws if the covariate has missing values (numerical h1 and twogroup h1)
			@test_throws Exception fetch!(SCP.ttest(l_job, "value3"; h1_missing=:error))
			@test_throws Exception fetch!(SCP.ttest(l_job, "group2"=>SCP.twogroup_covariate("A","B"); h1_missing=:error))
			# h0 missing is an error by default (h0_missing=:error), for numerical and categorical h0
			@test_throws Exception fetch!(SCP.ttest(l_job, "value"; h0="value3"))
			@test_throws Exception fetch!(SCP.ttest(l_job, "value"; h0="group2"))

			# :skip (default) excludes missing observations. numerical h1:
			mask = .!ismissing.(obs.value3)
			nzm = vec(any(!iszero, X[:,mask]; dims=2))
			gtT, gtP, gtβ = ttest_ground_truth(X[:,mask], obs[mask,:], "value3", ())
			r = fetch!(SCP.ttest(l_job, "value3"; do_sort=false))
			@test r.t[nzm] ≈ gtT[nzm]
			@test r.pValue[nzm] ≈ gtP[nzm]
			@test r.difference[nzm] ≈ gtβ[nzm]

			# twogroup h1 with missing labels: keeps only the A/B observations
			mask = isequal.(obs.group2, "A") .| isequal.(obs.group2, "B")
			nzm = vec(any(!iszero, X[:,mask]; dims=2))
			gtT, gtP, gtβ = ttest_ground_truth(X, obs, "group2", "A", "B", ())
			r = fetch!(SCP.ttest(l_job, "group2"=>SCP.twogroup_covariate("A","B"); do_sort=false))
			@test r.t[nzm] ≈ gtT[nzm]
			@test r.pValue[nzm] ≈ gtP[nzm]
			@test r.difference[nzm] ≈ gtβ[nzm]
		end

		@testset "column names" begin
			r = fetch!(SCP.ttest(l_job, "value"; statistic_col="my_t", pvalue_col="my_p", difference_col="my_d"))
			@test names(r) == [idcol, "my_t", "my_p", "my_d"]

			r = fetch!(SCP.ttest(l_job, "value"; statistic_col=nothing))
			@test names(r) == [idcol, "pValue", "difference"]

			r = fetch!(SCP.ttest(l_job, "value"; pvalue_col=nothing))
			@test names(r) == [idcol, "t", "difference"]

			r = fetch!(SCP.ttest(l_job, "value"; difference_col=nothing))
			@test names(r) == [idcol, "t", "pValue"]
		end

		@testset "var_cols" begin
			# extra var columns are carried through, inserted before the statistic columns
			r = fetch!(SCP.ttest(l_job, "value"; var_cols="name", do_sort=false))
			@test names(r) == [idcol, "name", "t", "pValue", "difference"]
			@test isequal(r.name, l.var.name)

			r = fetch!(SCP.ttest(l_job, "value"; var_cols=("name","feature_type"), do_sort=false))
			@test names(r) == [idcol, "name", "feature_type", "t", "pValue", "difference"]
		end

		@testset "center=false" begin
			# no intercept and no h0 (null model)
			gtT, gtP, gtβ = ttest_ground_truth(X, obs, "value", (); center=false)
			r = fetch!(SCP.ttest(l_job, "value"; center=false, do_sort=false))
			@test names(r) == [idcol, "t", "pValue", "difference"]
			@test r.t[nzmask] ≈ gtT[nzmask]
			@test r.pValue[nzmask] ≈ gtP[nzmask]
			@test r.difference[nzmask] ≈ gtβ[nzmask]
		end

		@testset "sorting" begin
			ru = fetch!(SCP.ttest(l_job, "value"; do_sort=false))
			rs = fetch!(SCP.ttest(l_job, "value")) # do_sort=true (default)
			@test isequal(ru[!, idcol], l.var[!, idcol])                          # unsorted keeps var order
			@test isequal(rs[!, idcol], ru[!, idcol][sortperm(abs.(ru.t); rev=true)]) # sorted by |t| desc
		end
	end
end
