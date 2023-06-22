function ttest_ground_truth(A, obs, formula)
	t = zeros(size(A,1))
	p = zeros(size(A,1))

	df = copy(obs)
	df.y = zeros(size(A,2))
	for i in 1:size(A,1)
		df.y = A[i,:]
		m = lm(formula, df)
		table = coeftable(m)

		t[i] = table.cols[table.teststatcol][end]
		p[i] = table.cols[table.pvalcol][end]
	end
	
	t,p
end
function ttest_ground_truth(A, obs, h1, h0::Tuple)
	h1 in h0 && return zeros(size(A,1)), ones(size(A,1))

	formula = _formula(h0..., h1)
	return ttest_ground_truth(A, obs, formula)
end


@testset "t-test" begin
	N = size(transformed,2)

	t = copy(transformed)
	t.obs.value2 = t.obs.value.^2

	# annotations with missing values
	t.obs.value3 = missings(Float64, size(t,2))
	t.obs.value3[1:2:end] .= 1:cld(size(t,2),2)
	t.obs.group2 = replace(t.obs.group, "C"=>missing)


	A = t.matrix*I(N)

	setup = (("value", (), "value_"),
             ("value", ("group",), "value_H0_group_"),
             ("value2", ("value",), "value2_H0_value_"),
             ("value2", ("group","value"), "value2_H0_group_value_"),
             ("value", ("value",), "value_H0_value_"),
            )

	@testset "H1:$h1, H0:$(join(h0,','))" for (h1,h0,prefix) in setup
		gtT, gtP = ttest_ground_truth(A, t.obs, h1, h0)

		@testset "$f" for f in (ttest_table, ttest, ttest!)
			data = f==ttest! ? copy(t) : t

			result = f(data, h1; h0)

			t_col = "t"
			p_col = "pValue"
			if f==ttest_table
				df = result
			else
				df = result.var
				t_col = string(prefix,t_col)
				p_col = string(prefix,p_col)

				# columns should only be added to source if using ttest!
				@test hasproperty(data.var, t_col) == (f==ttest!)
				@test hasproperty(data.var, p_col) == (f==ttest!)
			end

			@test df[:,t_col] ≈ gtT
			@test df[:,p_col] ≈ gtP
		end
	end

	@testset "Missing" begin
		@test_throws r"Missing values.+numerical" ttest_table(t, "value"; h0="value3")
		@test_throws r"Missing values.+categorical" ttest_table(t, "value"; h0="group2")
	end

	@testset "Column names" begin
		gtT, gtP = ttest_ground_truth(A, t.obs, "value", ())

		df = ttest_table(t, "value"; statistic_col="my_t", pvalue_col="my_p")
		@test df[:,"my_t"] ≈ gtT
		@test df[:,"my_p"] ≈ gtP

		df = ttest_table(t, "value"; statistic_col=nothing, pvalue_col="my_p")
		@test !hasproperty(df, "t")
		@test df[:,"my_p"] ≈ gtP

		df = ttest_table(t, "value"; statistic_col="my_t", pvalue_col=nothing)
		@test df[:,"my_t"] ≈ gtT
		@test !hasproperty(df, "pValue")

		data = ttest(t, "value"; statistic_col="my_t", pvalue_col="my_p")
		@test data.var[:,"my_t"] ≈ gtT
		@test data.var[:,"my_p"] ≈ gtP
		@test !hasproperty(t.var, "my_t")
		@test !hasproperty(t.var, "my_p")

		data = copy(t)
		ttest!(data, "value"; statistic_col="my_t", pvalue_col="my_p")
		@test data.var[:,"my_t"] ≈ gtT
		@test data.var[:,"my_p"] ≈ gtP


		data = ttest(t, "value"; prefix="another_")
		@test data.var[:,"another_t"] ≈ gtT
		@test data.var[:,"another_pValue"] ≈ gtP
		@test !hasproperty(t.var, "another_t")
		@test !hasproperty(t.var, "another_pValue")

		data = copy(t)
		ttest!(data, "value"; prefix="another_")
		@test data.var[:,"another_t"] ≈ gtT
		@test data.var[:,"another_pValue"] ≈ gtP
	end
end
