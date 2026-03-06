using SingleCellProjections
using .SingleCellProjections.SingleCellProjectionsCore
using Test
using StaticArrays
using Statistics


using .SingleCellProjectionsCore: BarnesHutTree, build!

@testset "SingleCellProjectionsCore.jl" begin
	# include("ranktests.jl")
	include("datamatrix.jl")
	# include("duplicate_var_ids.jl")
	# include("ftest_tests.jl")
	# include("ttest_tests.jl")
	# include("mannwhitney_tests.jl")
	include("test_barnes_hut.jl")
end
