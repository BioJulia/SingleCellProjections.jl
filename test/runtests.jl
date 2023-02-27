using SingleCellProjections
using Test
using LinearAlgebra
using SparseArrays
using StableRNGs
using StaticArrays
using Statistics
using DelimitedFiles
using CodecZlib
using SCTransform

import SingleCellProjections: BarnesHutTree, build!


# include("MatrixExpressions/runtests.jl")


@testset "SingleCellProjections.jl" begin
    include("basic.jl")
    # include("test_barnes_hut.jl")
end
