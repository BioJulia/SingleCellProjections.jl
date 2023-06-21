using SingleCellProjections
using .SingleCellProjections.MatrixExpressions
using Test
using LinearAlgebra
using SparseArrays
using Random
using StableRNGs
using StaticArrays
using Statistics
using HypothesisTests
using GLM
using DelimitedFiles
using DataFrames
using CodecZlib
using SCTransform

using UMAP
using TSne

using SingleCellProjections: BarnesHutTree, build!


include("MatrixExpressions/runtests.jl")


include("test_utils.jl")
include("common_data.jl")

@testset "SingleCellProjections.jl" begin
    include("ranktests.jl")
    include("datamatrix.jl")
    include("load.jl")
    include("basic.jl")
    include("annotate.jl")
    include("ftest_tests.jl")
    include("ttest_tests.jl")
    include("mannwhitney_tests.jl")
    include("projections.jl")
    include("test_barnes_hut.jl")
end
