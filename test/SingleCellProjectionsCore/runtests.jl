using SingleCellProjections
using .SingleCellProjections.SingleCellProjectionsCore
# using .SingleCellProjectionsCore.MatrixExpressions
using Test
using LinearAlgebra
using SparseArrays
using Random
using StableRNGs
using StaticArrays
using Statistics
using HypothesisTests
import GLM
using GLM: StatsModels, lm, coeftable, modelmatrix
using DelimitedFiles
using DataFrames
using CodecZlib
using SCTransform

using UMAP
using TSne
using PrincipalMomentAnalysis

using .SingleCellProjectionsCore: Annotations
using .SingleCellProjectionsCore: BarnesHutTree, build!, CovariateDesc, covariate_prefix

include("test_utils.jl")
include("common_data.jl")

@testset "SingleCellProjectionsCore.jl" begin
	include("ranktests.jl")
	include("datamatrix.jl")
	include("load.jl")
	include("basic.jl")
	include("duplicate_var_ids.jl")
	include("annotate.jl")
	include("ftest_tests.jl")
	include("ttest_tests.jl")
	include("mannwhitney_tests.jl")
	include("projections.jl")
	include("test_barnes_hut.jl")
end
