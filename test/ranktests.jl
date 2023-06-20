using SingleCellProjections: ustatistic_single, mannwhitney_single, mannwhitney_σ

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
end
