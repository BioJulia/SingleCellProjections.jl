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

@testset "FTest" begin
	P,N = (50,587)

	t = copy(transformed)
	t.obs.value2 = t.obs.value.^2

	A = t.matrix*I(N)

	@testset "categorical" begin
		df = ftest_table(t, "group")
		gtF, gtP = ftest_ground_truth(A, t.obs, @formula(y~1+group), @formula(y~1))

		@test df.F ≈ gtF
		@test df.pValue ≈ gtP
	end
end
