using .SCPCore: Blocks, blockify, row_sum_squared, ftest_table, ttest_table, implicitsvd
using DataFrames
using LinearAlgebra
using SparseArrays
using StableRNGs

# Products of a `Blocks` matrix (or its adjoint) with several row blocks and a dense matrix are
# themselves `Blocks`. Column blocks become row blocks in the adjoint, as used by `implicitsvd`.
function run_blocked_products_tests()
	@testset "Blocked products" begin
		P,N = 60,45
		X = sprand(StableRNG(1), P, N, 0.3)

		var = DataFrame("id"=>string.("v", 1:P))
		h0 = ones(N,1)
		groups = rand(StableRNG(2), 1:3, N)
		h1_cat = Float64[g==k for g in groups, k in 1:3]
		value = randn(StableRNG(3), N)
		h1_num = reshape(value .- mean(value), :, 1)

		@testset "$(size(Xb.blocks)) blocks" for Xb in (blockify(X; row_block_size=16, col_block_size=15),
		                                                blockify(X; row_block_size=P, col_block_size=15))
			@testset "ftest_table" begin
				kwargs = (; statistic_col="F", pvalue_col="pValue", do_sort=false)
				r = ftest_table(X, var, row_sum_squared(X), h1_cat, h0; kwargs...)
				rb = ftest_table(Xb, var, row_sum_squared(Xb), h1_cat, h0; kwargs...)
				@test rb.F ≈ r.F
				@test rb.pValue ≈ r.pValue
			end

			@testset "ttest_table" begin
				kwargs = (; statistic_col="t", pvalue_col="pValue", difference_col="difference", do_sort=false)
				r = ttest_table(X, var, row_sum_squared(X), h1_num, 1.0, h0; kwargs...)
				rb = ttest_table(Xb, var, row_sum_squared(Xb), h1_num, 1.0, h0; kwargs...)
				@test rb.t ≈ r.t
				@test rb.pValue ≈ r.pValue
				@test rb.difference ≈ r.difference
			end

			@testset "implicitsvd" begin
				F = implicitsvd(X; nsv=3, subspacedims=8, rng=StableRNG(4))
				Fb = implicitsvd(Xb; nsv=3, subspacedims=8, rng=StableRNG(4))
				@test Fb.S ≈ F.S
				@test Fb.U ≈ F.U
				@test Fb.Vt ≈ F.Vt
			end
		end
	end
end
