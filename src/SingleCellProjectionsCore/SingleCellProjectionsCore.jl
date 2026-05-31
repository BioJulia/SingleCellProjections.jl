module SingleCellProjectionsCore

using LinearAlgebra

using SparseArrays
using Statistics

using ReadOnlyArrays

using HDF5, H5Zblosc
using DataFrames

using Missings

using Random
using StableRNGs

using NearestNeighbors
using StaticArrays

using Distributions

import SCTransform: SCTransform, scparams, sctransform
using SingleCell10x

using ChunkSplitters
using OhMyThreads
using TaskLocalValues

using ProgressMeter: Progress, next!, finish! # For progress bars


include("../MatrixExpressions/MatrixExpressions.jl")
using .MatrixExpressions
using .MatrixExpressions: colsum, colsum!, rowsum, rowsum!


# # This symbol is only defined on Julia versions that support extensions
# isdefined(Base, :get_extension) || using Requires


# TODO: Is this still needed?
const Index = Union{AbstractVector, Colon}


include("blocks.jl")


include("random.jl")
include("table_utils.jl")
include("threaded_sparse_row_map.jl")

include("bilinear.jl")
include("sctransformsparse.jl")

include("implicitsvd.jl")

include("datamatrix.jl")
include("subset_expression.jl")

include("adjacency_matrices.jl")

include("barnes_hut.jl")
include("force_layout.jl")

include("h5ad.jl")

include("mannwhitney.jl")

include("filter.jl")
include("load.jl")
include("transform.jl")
include("sum_squared.jl")
include("design.jl")
include("normalize.jl")
include("reduce.jl")
include("statistical_tests.jl")
include("counts_fraction.jl")
include("counts_sum.jl")

include("annotation_transfer.jl")

include("local_outlier_factor.jl")

# include("precompile.jl")

end
