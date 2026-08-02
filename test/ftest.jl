using Test
using SingleCellProjections
import SingleCellProjections as SCP
using SingleCellProjections: SCPCore
using .SCPCore: unblockify
using ReproducibleJobs: fetch!
using DataFrames


function run_ftest_tests()
	@testset "F-test" begin
		counts_job = SCP.load_counts(h5_path; sample_names="a")
		counts_job = SCP.add_obs_column(counts_job, "group", counts_obs_group)
		counts_job = SCP.add_obs_column(counts_job, "value", counts_obs_value)
		counts_job = SCP.add_obs_column(counts_job, "value2", counts_obs_value.^2)
		# value3: numerical with missing values (fill odd indices only)
		value3 = missings(Float64, length(counts_obs_value))
		value3[1:2:end] .= 1:cld(length(counts_obs_value),2)
		counts_job = SCP.add_obs_column(counts_job, "value3", value3)
		counts_job = SCP.add_obs_column(counts_job, "group2", replace(counts_obs_group, "C"=>missing))
		l_job = SCP.logtransform(counts_job)

		l = fetch!(l_job)
		X = convert(Matrix{Float64}, unblockify(materialize(l)))
		obs = l.obs
		idcol = only(names(l.var, 1))
		nzmask = vec(any(!iszero, X; dims=2)) # exclude all-zero variables (F/p are NaN there)

		@testset "H1:$h1 H0:$h0" for (h1, h0) in (
				("group",            ()),
				("value",            ()),
				("group",            ("value",)),
				("value",            ("group",)),
				(("value","value2"), ()),
				(("value","value2"), ("group",)),
				("value2",           ("group","value")),
				("value",            ("value",)), # h1 ⊆ h0 -> F=0, p=1
			)
			gtF, gtP = ftest_ground_truth(X, obs, h1 isa Tuple ? h1 : (h1,), h0)
			r = fetch!(SCP.ftest(l_job, h1; h0, do_sort=false))
			@test names(r) == [idcol, "F", "pValue"]
			@test r.F[nzmask] ≈ gtF[nzmask]
			@test r.pValue[nzmask] ≈ gtP[nzmask]
		end

		@testset "missing handling" begin
			# :error throws if the covariate has missing values
			@test_throws Exception fetch!(SCP.ftest(l_job, "value3"; h1_missing=:error))
			@test_throws Exception fetch!(SCP.ftest(l_job, "group2"; h1_missing=:error))
			# h0 missing is an error by default (h0_missing=:error)
			@test_throws Exception fetch!(SCP.ftest(l_job, "value"; h0="value3"))
			@test_throws Exception fetch!(SCP.ftest(l_job, "value"; h0="group2"))

			# :skip (default) excludes the missing observations. The mask reduces the cells, so
			# recompute the non-zero-variable mask on the filtered subset.
			for col in ("value3", "group2")
				mask = .!ismissing.(obs[!, col])
				nzm = vec(any(!iszero, X[:,mask]; dims=2))
				gtF, gtP = ftest_ground_truth(X[:,mask], obs[mask,:], (col,), ())
				r = fetch!(SCP.ftest(l_job, col; do_sort=false))
				@test r.F[nzm] ≈ gtF[nzm]
				@test r.pValue[nzm] ≈ gtP[nzm]
			end
		end

		@testset "column names" begin
			r = fetch!(SCP.ftest(l_job, "group"; statistic_col="my_F", pvalue_col="my_p"))
			@test names(r) == [idcol, "my_F", "my_p"]

			r = fetch!(SCP.ftest(l_job, "group"; statistic_col=nothing))
			@test names(r) == [idcol, "pValue"]

			r = fetch!(SCP.ftest(l_job, "group"; pvalue_col=nothing))
			@test names(r) == [idcol, "F"]
		end

		@testset "var_cols" begin
			# extra var columns are carried through, inserted before the statistic columns
			r = fetch!(SCP.ftest(l_job, "group"; var_cols="name", do_sort=false))
			@test names(r) == [idcol, "name", "F", "pValue"]
			@test isequal(r.name, l.var.name)

			r = fetch!(SCP.ftest(l_job, "group"; var_cols=("name","feature_type"), do_sort=false))
			@test names(r) == [idcol, "name", "feature_type", "F", "pValue"]
		end

		@testset "center=false" begin
			# no intercept and no h0 (null model)
			gtF, gtP = ftest_ground_truth(X, obs, ("value",), (); center=false)
			r = fetch!(SCP.ftest(l_job, "value"; center=false, do_sort=false))
			@test names(r) == [idcol, "F", "pValue"]
			@test r.F[nzmask] ≈ gtF[nzmask]
			@test r.pValue[nzmask] ≈ gtP[nzmask]
		end

		@testset "sorting" begin
			ru = fetch!(SCP.ftest(l_job, "group"; do_sort=false))
			rs = fetch!(SCP.ftest(l_job, "group")) # do_sort=true (default)
			@test isequal(ru[!, idcol], l.var[!, idcol])              # unsorted keeps var order
			@test isequal(rs[!, idcol], ru[!, idcol][sortperm(ru.F; rev=true)]) # sorted by F desc
		end
	end
end
