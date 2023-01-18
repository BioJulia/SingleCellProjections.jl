using SingleCellProjections
using Test
using StableRNGs
using StaticArrays
using Statistics

import SingleCellProjections: BarnesHutTree, build!


include("MatrixExpressions/runtests.jl")


@testset "SingleCellProjections.jl" begin
    include("test_barnes_hut.jl")
end
