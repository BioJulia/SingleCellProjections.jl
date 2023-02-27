using SingleCellProjections
using Test
using SparseArrays
using StableRNGs
using StaticArrays
using Statistics
using DelimitedFiles
using CodecZlib

import SingleCellProjections: BarnesHutTree, build!


include("MatrixExpressions/runtests.jl")


@testset "SingleCellProjections.jl" begin
    include("basic.jl")
    include("test_barnes_hut.jl")
end
