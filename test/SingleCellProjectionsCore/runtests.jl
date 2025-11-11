using SingleCellProjections
using .SingleCellProjections.SingleCellProjectionsCore
# using .SingleCellProjectionsCore.MatrixExpressions
using Test
# using Random
using StaticArrays
using Statistics
# using SCTransform


# using .SingleCellProjectionsCore: Annotations
using .SingleCellProjectionsCore: BarnesHutTree, build!, CovariateDesc, covariate_prefix

@testset "SingleCellProjectionsCore.jl" begin
	# include("ranktests.jl")
	# include("datamatrix.jl")
	# include("load.jl")
	# include("basic.jl")
	# include("duplicate_var_ids.jl")
	# include("annotate.jl")
	# include("ftest_tests.jl")
	# include("ttest_tests.jl")
	# include("mannwhitney_tests.jl")
	# include("projections.jl")
	include("test_barnes_hut.jl")
end
