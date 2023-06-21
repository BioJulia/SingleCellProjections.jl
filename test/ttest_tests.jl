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
function ttest_ground_truth(A, obs, test, null::Tuple)
	test in null && return zeros(size(A,1)), ones(size(A,1))

	formula = _formula(null..., test)
	return ttest_ground_truth(A, obs, formula)
end


@testset "t-Test" begin
	N = size(transformed,2)

	t = copy(transformed)
	t.obs.value2 = t.obs.value.^2

	A = t.matrix*I(N)

	setup = (("value", ()),
             ("value", ("group",)),
             ("value2", ("value",)),
             ("value2", ("group","value")),
             ("value", ("value",)),
            )

	@testset "H1:$test, H0:$(join(null,','))" for (test,null) in setup
		df = ttest_table(t, test; null)
		gtt, gtP = ttest_ground_truth(A, t.obs, test, null)

		@test df.t ≈ gtt
		@test df.pValue ≈ gtP
	end
end
