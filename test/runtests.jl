using SingleCellProjections
using .SingleCellProjections.MatrixExpressions
using Test
using LinearAlgebra
using SparseArrays
using Random
using StableRNGs
using StaticArrays
using Statistics
using DelimitedFiles
using DataFrames
using CodecZlib
using SCTransform

using UMAP
using TSne

import SingleCellProjections: BarnesHutTree, build!


include("MatrixExpressions/runtests.jl")


include("test_utils.jl")
include("common_data.jl")

@testset "SingleCellProjections.jl" begin
    include("datamatrix.jl")
    include("load.jl")
    include("basic.jl")
    include("annotate.jl")
    include("projections.jl")
    include("test_barnes_hut.jl")
end
