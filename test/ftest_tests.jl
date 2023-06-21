function ftest_ground_truth(A, obs, h1_formula, h0_formula)
	F = zeros(size(A,1))
	p = zeros(size(A,1))

	df = copy(obs)
	df.y = zeros(size(A,2))
	for i in 1:size(A,1)
		df.y = A[i,:]
		h0 = lm(h0_formula, df)
		h1 = lm(h1_formula, df)
		ft = ftest(h0.model, h1.model)
		F[i] = ft.fstat[end]
		p[i] = ft.pval[end]
	end
	
	F,p
end
function ftest_ground_truth(A, obs, test::Tuple, null::Tuple)
	all(in(null), test) && return zeros(size(A,1)), ones(size(A,1))

	h1_formula = _formula(null..., test...)
	h0_formula = _formula(null...)
	return ftest_ground_truth(A, obs, h1_formula, h0_formula)
end


@testset "F-Test" begin
	N = size(transformed,2)

	t = copy(transformed)
	t.obs.value2 = t.obs.value.^2

	A = t.matrix*I(N)

	setup = ((("group",), ()),
             (("value",), ()),
             (("group",), ("value",)),
             (("value",), ("group",)),
             (("value","value2"), ()),
             (("value","value2"), ("group",)),
             (("value2",), ("group","value")),
             (("value",), ("value",)),
            )

	@testset "H1:$(join(test,',')), H0:$(join(null,','))" for (test,null) in setup
		df = ftest_table(t, test; null)
		gtF, gtP = ftest_ground_truth(A, t.obs, test, null)

		@test df.F ≈ gtF
		@test df.pValue ≈ gtP
	end

	# @testset "categorical" begin
	# 	df = ftest_table(t, "group")
	# 	gtF, gtP = ftest_ground_truth(A, t.obs, @formula(y~1+group), @formula(y~1))

	# 	@test df.F ≈ gtF
	# 	@test df.pValue ≈ gtP
	# end
end
