using SingleCellProjections: ustatistic_single, mannwhitney_single, mannwhitney_σ, mannwhitney_sparse

@testset "MannWhitney" begin
	@testset "No ties" begin
		X = sparse([3; 2; 10; 9; 5; 6; 7; 1; 8; 4;;])

		@testset "Balanced" begin
			g = [1,2,1,2,1,2,1,2,1,2]

			mw = ApproximateMannWhitneyUTest(X[g.==1], X[g.==2]) # ground truth
			U,t_adj = ustatistic_single(X,1,g,5,5)
			σ = mannwhitney_σ(5,5,t_adj)
			U2,p = mannwhitney_single(X,1,g,5,5)

			@test U==U2
			@test U == mw.U
			@test t_adj == 0
			@test σ ≈ mw.sigma
			@test p ≈ pvalue(mw)
		end

		@testset "Unbalanced" begin
			g = [1,2,1,2,1,2,1,1,1,1]
			mw = ApproximateMannWhitneyUTest(X[g.==1], X[g.==2]) # ground truth
			U,t_adj = ustatistic_single(X,1,g,7,3)
			σ = mannwhitney_σ(7,3,t_adj)
			U2,p = mannwhitney_single(X,1,g,7,3)

			@test U==U2
			@test U == mw.U
			@test t_adj == 0
			@test σ ≈ mw.sigma
			@test p ≈ pvalue(mw)
		end

		@testset "Skipped" begin
			g = [1,2,2,0,1,2,1,0,1,1]
			mw = ApproximateMannWhitneyUTest(X[g.==1], X[g.==2]) # ground truth
			U,t_adj = ustatistic_single(X,1,g,5,3)
			σ = mannwhitney_σ(5,3,t_adj)
			U2,p = mannwhitney_single(X,1,g,5,3)

			@test U==U2
			@test U == mw.U
			@test t_adj == 0
			@test σ ≈ mw.sigma
			@test p ≈ pvalue(mw)
		end
	end

	@testset "Ties" begin
		X = sparse([3; 3; 1; 3; 1; 2; 4; 5; 3; 2;;])

		@testset "Balanced" begin
			g = [1,2,1,2,1,2,1,2,1,2]
			mw = ApproximateMannWhitneyUTest(X[g.==1], X[g.==2]) # ground truth
			U,t_adj = ustatistic_single(X,1,g,5,5)
			σ = mannwhitney_σ(5,5,t_adj)
			U2,p = mannwhitney_single(X,1,g,5,5)

			@test U==U2
			@test U == mw.U
			@test t_adj == mw.tie_adjustment
			@test σ ≈ mw.sigma
			@test p ≈ pvalue(mw)
		end
		
		@testset "Unbalanced" begin
			g = [1,2,1,2,1,2,1,1,1,1]
			mw = ApproximateMannWhitneyUTest(X[g.==1], X[g.==2]) # ground truth
			U,t_adj = ustatistic_single(X,1,g,7,3)
			σ = mannwhitney_σ(7,3,t_adj)
			U2,p = mannwhitney_single(X,1,g,7,3)

			@test U==U2
			@test U == mw.U
			@test t_adj == mw.tie_adjustment
			@test σ ≈ mw.sigma
			@test p ≈ pvalue(mw)
		end
		
		@testset "Skipped" begin
			g = [1,2,2,0,1,2,1,0,1,1]
			mw = ApproximateMannWhitneyUTest(X[g.==1], X[g.==2]) # ground truth
			U,t_adj = ustatistic_single(X,1,g,5,3)
			σ = mannwhitney_σ(5,3,t_adj)
			U2,p = mannwhitney_single(X,1,g,5,3)

			@test U==U2
			@test U == mw.U
			@test t_adj == mw.tie_adjustment
			@test σ ≈ mw.sigma
			@test p ≈ pvalue(mw)
		end
	end

	@testset "Zeros" begin
		X = sparse([1; 5; 3; 3; 0; 1; 1; 0; 0; 0; 5; 1; 2; 0; 0; 2; 0;;])
		g =        [0, 1, 1, 2, 2, 1, 2, 2, 2, 0, 1, 2, 0, 2, 1, 0, 0]

		mw = ApproximateMannWhitneyUTest(X[g.==1], X[g.==2]) # ground truth
		U,t_adj = ustatistic_single(X,1,g,5,7)
		σ = mannwhitney_σ(5,7,t_adj)
		U2,p = mannwhitney_single(X,1,g,5,7)

		@test U==U2
		@test U == mw.U
		@test t_adj == mw.tie_adjustment
		@test σ ≈ mw.sigma
		@test p ≈ pvalue(mw)
	end

	@testset "Empty" begin
		X = sparse([3; 2; 6; 9; 2; 6; 7; 1; 8; 4;;])
		g1 =       [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
		g2 =       [0, 2, 0, 2, 2, 0, 0, 0, 0, 0]
		g3 =       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

		U1,t_adj1 = ustatistic_single(X,1,g1,4,0)
		@test U1 == 0.0
		@test t_adj1 == 0.0
		U1,p1 = mannwhitney_single(X,1,g1,4,0)
		@test U1 == 0.0
		@test p1 == 1.0

		U2,t_adj2 = ustatistic_single(X,1,g2,0,3)
		@test U2 == 0.0
		@test t_adj2 == 2^3-2
		U2,p2 = mannwhitney_single(X,1,g2,0,3)
		@test U2 == 0.0
		@test p2 == 1.0

		U3,t_adj3 = ustatistic_single(X,1,g3,0,0)
		@test U3 == 0.0
		@test t_adj3 == 0.0
		U3,p3 = mannwhitney_single(X,1,g3,0,0)
		@test U3 == 0.0
		@test p3 == 1.0
	end

	@testset "Matrix" begin
		g = rand(StableRNG(700), 0:2, 100)
		n0 = count(==(0),g)
		n1 = count(==(1),g)
		n2 = count(==(2),g)

		# Generate a matrix with different properties for different variables
		# To get very different number of zeros and very different p-values
		X = zeros(20,100)
		rng = StableRNG(600)
		for i in 1:size(X,1)
			μ0,σ0 = 0,1
			μ1,σ1 = rand(rng,-5:5), rand(rng,-5:5)
			μ2,σ2 = rand(rng,-5:5), rand(rng,-5:5)
			X[i,g.==0] .= round.(max.(0.0, randn(n0).*σ0.+μ0); digits=1)
			X[i,g.==1] .= round.(max.(0.0, randn(n1).*σ1.+μ1); digits=1)
			X[i,g.==2] .= round.(max.(0.0, randn(n2).*σ2.+μ2); digits=1)
		end

		Xs = sparse(X)
		U,p = mannwhitney_sparse(Xs,g; nworkers=1, chunk_size=100) # non-threaded, one chunk
		U2,p2 = mannwhitney_sparse(Xs,g; nworkers=3, chunk_size=4) # threaded, multiple chunks

		for i in 1:size(X,1)
			mw = ApproximateMannWhitneyUTest(X[i,g.==1], X[i,g.==2])

			@test U[i] == mw.U
			@test p[i] ≈ pvalue(mw)

			@test U2[i] == mw.U
			@test p2[i] ≈ pvalue(mw)
		end
	end
end
