function ftest_ground_truth(A, obs, h1_formula, h0_formula)
	F = zeros(size(A,1))
	p = zeros(size(A,1))

	df = copy(obs)
	df.y = zeros(size(A,2))
	for i in 1:size(A,1)
		df.y = A[i,:]
		h0 = lm(h0_formula, df)
		h1 = lm(h1_formula, df)
		ft = GLM.ftest(h0.model, h1.model)
		F[i] = ft.fstat[end]
		p[i] = ft.pval[end]
	end
	
	F,p
end
function ftest_ground_truth(A, obs, h1::Tuple, h0::Tuple)
	# simple unwrapping of Covariates, does not care about types or two-groups
	h1 = (x->x isa CovariateDesc ? x.name : x).(h1)
	h0 = (x->x isa CovariateDesc ? x.name : x).(h0)

	all(in(h0), h1) && return zeros(size(A,1)), ones(size(A,1))

	h1_formula = _formula(h0..., h1...)
	h0_formula = _formula(h0...)
	return ftest_ground_truth(A, obs, h1_formula, h0_formula)
end


@testset "F-test" begin
	N = size(transformed,2)

	t = copy(transformed)
	t.obs.value2 = t.obs.value.^2

	# annotations with missing values
	t.obs.value3 = missings(Float64, size(t,2))
	t.obs.value3[1:2:end] .= 1:cld(size(t,2),2)
	t.obs.group2 = replace(t.obs.group, "C"=>missing)

	A = t.matrix*I(N)

	setup = ((("group",), (), "group_"),
             (("value",), (), "value_"),
             (("group",), ("value",), "group_H0_value_"),
             (("value",), ("group",), "value_H0_group_"),
             (("value","value2"), (), "value_value2_"),
             (("value","value2"), ("group",), "value_value2_H0_group_"),
             (("value2",), ("group","value"), "value2_H0_group_value_"),
             (("value",), ("value",), "value_H0_value_"),
             ((covariate("value"),), (), "value_"),
             ((covariate("group"),), (), "group_"),
             (("group",), (covariate("value"),), "group_H0_value_"),
             (("value",), (covariate("group"),), "value_H0_group_"),
            )

	cov_str(c::CovariateDesc) = covariate_prefix(c,"'")
	cov_str(x) = x

	@testset "H1:$(join(cov_str.(h1),',')), H0:$(join(cov_str.(h0),','))" for (h1,h0,prefix) in setup
		gtF, gtP = ftest_ground_truth(A, t.obs, h1, h0)

		@testset "$f" for f in (ftest_table, ftest, ftest!)
			data = f==ftest! ? copy(t) : t

			result = f(data, h1; h0)

			f_col = "F"
			p_col = "pValue"
			if f==ftest_table
				df = result
			else
				df = result.var
				f_col = string(prefix,f_col)
				p_col = string(prefix,p_col)

				# columns should only be added to source if using ftest!
				@test hasproperty(data.var, f_col) == (f==ftest!)
				@test hasproperty(data.var, p_col) == (f==ftest!)
			end

			@test df[:,f_col] ≈ gtF
			@test df[:,p_col] ≈ gtP
		end
	end

	@testset "Missing" begin
		@test_throws r"Missing values.+numerical" ftest_table(t, "value3"; h1_missing=:error)
		@test_throws r"Missing values.+categorical" ftest_table(t, "group2"; h1_missing=:error)
		@test_throws r"Missing values.+numerical" ftest_table(t, "value"; h0="value3")
		@test_throws r"Missing values.+categorical" ftest_table(t, "value"; h0="group2")

		mask = t.obs.value3 .!== missing
		gtF, gtP = ftest_ground_truth(A[:,mask], t.obs[mask,:], ("value3",), ())
		df = ftest_table(t, "value3"; h1_missing=:skip)
		@test df.F ≈ gtF
		@test df.pValue ≈ gtP
		gtF, gtP = ftest_ground_truth(A[:,mask], t.obs[mask,:], ("value3",), ("group",))
		df = ftest_table(t, "value3"; h0="group", h1_missing=:skip)
		@test df.F ≈ gtF
		@test df.pValue ≈ gtP

		mask = t.obs.group2 .!== missing
		gtF, gtP = ftest_ground_truth(A[:,mask], t.obs[mask,:], ("group2",), ())
		df = ftest_table(t, "group2"; h1_missing=:skip)
		@test df.F ≈ gtF
		@test df.pValue ≈ gtP
		gtF, gtP = ftest_ground_truth(A[:,mask], t.obs[mask,:], ("group2",), ("value",))
		df = ftest_table(t, "group2"; h0="value", h1_missing=:skip)
		@test df.F ≈ gtF
		@test df.pValue ≈ gtP
	end

	@testset "Column names" begin
		gtF, gtP = ftest_ground_truth(A, t.obs, ("group",), ())

		df = ftest_table(t, "group"; statistic_col="my_F", pvalue_col="my_p")
		@test df[:,"my_F"] ≈ gtF
		@test df[:,"my_p"] ≈ gtP

		df = ftest_table(t, "group"; statistic_col=nothing, pvalue_col="my_p")
		@test !hasproperty(df, "F")
		@test df[:,"my_p"] ≈ gtP

		df = ftest_table(t, "group"; statistic_col="my_F", pvalue_col=nothing)
		@test df[:,"my_F"] ≈ gtF
		@test !hasproperty(df, "pValue")

		data = ftest(t, "group"; statistic_col="my_F", pvalue_col="my_p")
		@test data.var[:,"my_F"] ≈ gtF
		@test data.var[:,"my_p"] ≈ gtP
		@test !hasproperty(t.var, "my_F")
		@test !hasproperty(t.var, "my_p")

		data = copy(t)
		ftest!(data, "group"; statistic_col="my_F", pvalue_col="my_p")
		@test data.var[:,"my_F"] ≈ gtF
		@test data.var[:,"my_p"] ≈ gtP

		data = ftest(t, "group"; prefix="another_")
		@test data.var[:,"another_F"] ≈ gtF
		@test data.var[:,"another_pValue"] ≈ gtP
		@test !hasproperty(t.var, "another_F")
		@test !hasproperty(t.var, "another_pValue")

		data = copy(t)
		ftest!(data, "group"; prefix="another_")
		@test data.var[:,"another_F"] ≈ gtF
		@test data.var[:,"another_pValue"] ≈ gtP
	end


	# @testset "categorical" begin
	# 	df = ftest_table(t, "group")
	# 	gtF, gtP = ftest_ground_truth(A, t.obs, @formula(y~1+group), @formula(y~1))

	# 	@test df.F ≈ gtF
	# 	@test df.pValue ≈ gtP
	# end
end
