function ttest_ground_truth(A, obs, formula, group_a)
	t = zeros(size(A,1))
	p = zeros(size(A,1))
	β = zeros(size(A,1))

	df = copy(obs)
	df.y = zeros(size(A,2))
	for i in 1:size(A,1)
		df.y = A[i,:]
		m = lm(formula, df)
		table = coeftable(m)
		@assert table.colnms[1] == "Coef."

		t[i] = table.cols[table.teststatcol][end]
		p[i] = table.cols[table.pvalcol][end]
		β[i] = table.cols[1][end]

		# a little hack since we cannot control which group is outputted by lm/coeftable
		if group_a !== nothing && !endswith(table.rownms[end], ": $group_a")
			t[i] = -t[i]
			β[i] = -β[i]
		end
	end
	
	t,p,β
end
function ttest_ground_truth(A, obs, h1, group_a, group_b, h0::Tuple)
	h1 in h0 && return zeros(size(A,1)), ones(size(A,1)), zeros(size(A,1))

	if !(eltype(obs[!,h1]) <: Union{Missing,Number})
		if group_a === nothing
			group_a = first(sort(obs[!,h1]))
		elseif group_b === nothing # overwrite everything except a
			obs = copy(obs)
			obs[.!isequal.(obs[!,h1],group_a), h1] .= "Not_$group_a"
		end
	end

	formula = _formula(h0..., h1)
	return ttest_ground_truth(A, obs, formula, group_a)
end
ttest_ground_truth(A, obs, h1, group_a, h0::Tuple) = ttest_ground_truth(A,obs,h1,group_a,nothing,h0)
ttest_ground_truth(A, obs, h1, h0::Tuple) = ttest_ground_truth(A,obs,h1,nothing,nothing,h0)


@testset "t-test" begin
	N = size(transformed,2)

	t = copy(transformed)
	t.obs.value2 = t.obs.value.^2

	# annotations with missing values
	t.obs.value3 = missings(Float64, size(t,2))
	t.obs.value3[1:2:end] .= 1:cld(size(t,2),2)
	t.obs.group2 = replace(t.obs.group, "C"=>missing)
	t.obs.twogroup = replace(t.obs.group, "C"=>"A")


	A = t.matrix*I(N)

	setup = ((("value",), (), "value_"),
             (("value",), ("group",), "value_H0_group_"),
             (("value2",), ("value",), "value2_H0_value_"),
             (("value2",), ("group","value"), "value2_H0_group_value_"),
             (("value",), ("value",), "value_H0_value_"),
             (("twogroup",), (), "twogroup_"),
             (("twogroup","A","B"), (), "twogroup_A_vs_B_"),
             (("twogroup","B","A"), (), "twogroup_B_vs_A_"),
             (("group","C"), (), "group_C_"),
            )

	h1_str(h1) = h1
	h1_str(h1,groupA) = "$(h1)_$groupA"
	h1_str(h1,groupA,groupB) = "$(h1)_$(groupA)_vs_$groupB"
	@testset "H1:$(h1_str(h1...)), H0:$(join(h0,','))" for (h1,h0,prefix) in setup
		gtT, gtP, gtβ = ttest_ground_truth(A, t.obs, h1..., h0)

		@testset "$f" for f in (ttest_table, ttest, ttest!)
			data = f==ttest! ? copy(t) : t

			result = f(data, h1...; h0)

			t_col = "t"
			p_col = "pValue"
			d_col = "difference"
			if f==ttest_table
				df = result
			else
				df = result.var
				t_col = string(prefix,t_col)
				p_col = string(prefix,p_col)
				d_col = string(prefix,d_col)

				# columns should only be added to source if using ttest!
				@test hasproperty(data.var, t_col) == (f==ttest!)
				@test hasproperty(data.var, p_col) == (f==ttest!)
				@test hasproperty(data.var, d_col) == (f==ttest!)
			end

			@test df[:,t_col] ≈ gtT
			@test df[:,p_col] ≈ gtP
			@test df[:,d_col] ≈ gtβ
		end
	end

	@testset "Missing" begin
		@test_throws r"Missing values.+numerical" ttest_table(t, "value3"; h1_missing=:error)
		@test_throws r"Missing values.+two-group" ttest_table(t, "group2"; h1_missing=:error)
		@test_throws r"Missing values.+numerical" ttest_table(t, "value"; h0="value3")
		@test_throws r"Missing values.+categorical" ttest_table(t, "value"; h0="group2")

		mask = t.obs.value3 .!== missing
		gtT, gtP = ttest_ground_truth(A[:,mask], t.obs[mask,:], "value3", ())
		df = ttest_table(t, "value3")
		@test df.t ≈ gtT
		@test df.pValue ≈ gtP
		gtT, gtP = ttest_ground_truth(A[:,mask], t.obs[mask,:], "value3", ("group",))
		df = ttest_table(t, "value3"; h0="group")
		@test df.t ≈ gtT
		@test df.pValue ≈ gtP

		mask = t.obs.group2 .!== missing
		gtT, gtP = ttest_ground_truth(A[:,mask], t.obs[mask,:], "group2", ())
		df = ttest_table(t, "group2")
		@test df.t ≈ gtT
		@test df.pValue ≈ gtP
		gtT, gtP = ttest_ground_truth(A[:,mask], t.obs[mask,:], "group2", ("value",))
		df = ttest_table(t, "group2"; h0="value")
		@test df.t ≈ gtT
		@test df.pValue ≈ gtP
	end

	@testset "Column names" begin
		gtT, gtP, gtβ = ttest_ground_truth(A, t.obs, "value", ())

		df = ttest_table(t, "value"; statistic_col="my_t", pvalue_col="my_p", difference_col="my_d")
		@test df[:,"my_t"] ≈ gtT
		@test df[:,"my_p"] ≈ gtP
		@test df[:,"my_d"] ≈ gtβ

		df = ttest_table(t, "value"; statistic_col=nothing, pvalue_col="my_p", difference_col="my_d")
		@test !hasproperty(df, "t")
		@test df[:,"my_p"] ≈ gtP
		@test df[:,"my_d"] ≈ gtβ

		df = ttest_table(t, "value"; statistic_col="my_t", pvalue_col=nothing, difference_col="my_d")
		@test df[:,"my_t"] ≈ gtT
		@test !hasproperty(df, "pValue")
		@test df[:,"my_d"] ≈ gtβ

		df = ttest_table(t, "value"; statistic_col="my_t", pvalue_col="my_p", difference_col=nothing)
		@test df[:,"my_t"] ≈ gtT
		@test df[:,"my_p"] ≈ gtP
		@test !hasproperty(df, "difference")

		data = ttest(t, "value"; statistic_col="my_t", pvalue_col="my_p", difference_col="my_d")
		@test data.var[:,"my_t"] ≈ gtT
		@test data.var[:,"my_p"] ≈ gtP
		@test data.var[:,"my_d"] ≈ gtβ
		@test !hasproperty(t.var, "my_t")
		@test !hasproperty(t.var, "my_p")
		@test !hasproperty(t.var, "my_d")

		data = copy(t)
		ttest!(data, "value"; statistic_col="my_t", pvalue_col="my_p", difference_col="my_d")
		@test data.var[:,"my_t"] ≈ gtT
		@test data.var[:,"my_p"] ≈ gtP
		@test data.var[:,"my_d"] ≈ gtβ


		data = ttest(t, "value"; prefix="another_")
		@test data.var[:,"another_t"] ≈ gtT
		@test data.var[:,"another_pValue"] ≈ gtP
		@test data.var[:,"another_difference"] ≈ gtβ
		@test !hasproperty(t.var, "another_t")
		@test !hasproperty(t.var, "another_pValue")
		@test !hasproperty(t.var, "another_difference")

		data = copy(t)
		ttest!(data, "value"; prefix="another_")
		@test data.var[:,"another_t"] ≈ gtT
		@test data.var[:,"another_pValue"] ≈ gtP
		@test data.var[:,"another_difference"] ≈ gtβ
	end
end
