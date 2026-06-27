using SingleCellProjections
using .SingleCellProjections.SCPCore
using Test
using StaticArrays
using Statistics


using .SCPCore: BarnesHutTree, build!

include("datamatrix.jl")
include("test_barnes_hut.jl")

function run_core_tests()
	@testset "SCPCore.jl" begin
		# include("ranktests.jl")
		run_datamatrix_tests()
		# include("duplicate_var_ids.jl")
		# include("ftest_tests.jl")
		# include("ttest_tests.jl")
		# include("mannwhitney_tests.jl")
		run_barnes_hut_tests()
	end
end
