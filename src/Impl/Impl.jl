module Impl

# Reference to the parent `SingleCellProjections` module, so that internal code can call
# the public API functions (e.g. `SCP.annotate_var`) that are defined in the parent.
const SCP = parentmodule(@__MODULE__)

import SCTransform
import SingleCell10x
using DataFrames
import StableHashTraits
import LinearAlgebra
using SparseArrays: sparse, SparseMatrixCSC
import StatsBase
using Statistics: mean

using ChunkSplitters
using OhMyThreads

using ReproducibleJobs
using ReproducibleJobs: create_job, fetched, prefetched, cached, throw_if_cancelled, ChecksummedFilePath, AbstractPreprocess, Preprocess, Preprocessing, CompoundResult, ProgressBar, _get_kwarg, checksummedfilepath_job, ifelse_job, ROArray, ROVec, ROMat, ROBitArray, ROBitVec, ROBitMat, TypeTag

using ReadOnlyArrays: ReadOnlyVector

using StyledStrings

import ..SingleCellProjectionsCore as SCPCore
using ..SingleCellProjectionsCore: DataMatrix, Blocks


# TODO: Where to put these?
ReproducibleJobs.deduplicate_type(::Type{<:LinearAlgebra.SVD}) = true
ReproducibleJobs.deconstruct_type(::Type{<:LinearAlgebra.SVD}) = true
ReproducibleJobs.type_to_tag(::Type{<:LinearAlgebra.SVD}) = TypeTag(:SVD)
ReproducibleJobs.tag_to_type(::Val{:SVD}) = LinearAlgebra.SVD
ReproducibleJobs.deconstruct(F::LinearAlgebra.SVD) = (F.U, F.S, F.Vt)
ReproducibleJobs.reconstruct(::Type{<:LinearAlgebra.SVD}, (U,S,Vt)::Tuple) = LinearAlgebra.SVD(U, S, Vt)


ReproducibleJobs.deduplicate_type(::Type{<:DataMatrix}) = true
ReproducibleJobs.deconstruct_type(::Type{<:DataMatrix}) = true
ReproducibleJobs.type_to_tag(::Type{<:DataMatrix}) = TypeTag(:DataMatrix)
ReproducibleJobs.tag_to_type(::Val{:DataMatrix}) = DataMatrix
ReproducibleJobs.deconstruct(data::DataMatrix{T}) where T = (data.matrix, data.var, data.obs)
ReproducibleJobs.reconstruct(::Type{<:DataMatrix}, (matrix,var,obs)::Tuple{T,DataFrame,DataFrame}) where T =
	DataMatrix(matrix, var, obs; duplicate_var=:ignore, duplicate_obs=:ignore) # Avoid doing validation when reconstructing!


# TODO: Where to put these?
ReproducibleJobs.deduplicate_type(::Type{<:Blocks}) = true
ReproducibleJobs.deconstruct_type(::Type{<:Blocks}) = true
ReproducibleJobs.type_to_tag(::Type{<:Blocks}) = TypeTag(:Blocks)
ReproducibleJobs.tag_to_type(::Val{:Blocks}) = Blocks
ReproducibleJobs.deconstruct(b::Blocks{T}) where T = (b.blocks,)

# TODO: These should maybe ensure we don't have ReadOnlyMatrices as matrix entries either
ReproducibleJobs.reconstruct(::Type{<:Blocks}, (blocks,)::Tuple{ROMat{T}}) where T = Blocks(parent(blocks))
ReproducibleJobs.reconstruct(::Type{<:Blocks}, (blocks,)::Tuple{T}) where T = Blocks(blocks)



# TODO: Where to put these?
ReproducibleJobs.deduplicate_type(::Type{<:SCPCore.AbstractCovariateDesc}) = false
ReproducibleJobs.deconstruct_weak_rec(x::T) where T<:SCPCore.AbstractCovariateDesc = x
ReproducibleJobs.reconstruct_weak_rec(x::T) where T<:SCPCore.AbstractCovariateDesc = x


# TODO: Where to put these?
ReproducibleJobs.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.MatrixRef}) = true
ReproducibleJobs.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.MatrixRef}) = true
ReproducibleJobs.type_to_tag(::Type{<:SCPCore.MatrixExpressions.MatrixRef}) = TypeTag(:ME_MatrixRef)
ReproducibleJobs.tag_to_type(::Val{:ME_MatrixRef}) = SCPCore.MatrixExpressions.MatrixRef
ReproducibleJobs.deconstruct(A::SCPCore.MatrixExpressions.MatrixRef{T}) where T = (A.name, A.matrix)
ReproducibleJobs.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixRef}, (name,matrix)::Tuple{Symbol,T}) where T =
	SCPCore.MatrixExpressions.MatrixRef(name, matrix)
ReproducibleJobs.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixRef}, (name,matrix)::Tuple{Symbol,ROArray{T}}) where T =
	SCPCore.MatrixExpressions.MatrixRef(name, parent(matrix))

ReproducibleJobs.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}) = true
ReproducibleJobs.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}) = true
ReproducibleJobs.type_to_tag(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}) = TypeTag(:ME_MatrixProduct)
ReproducibleJobs.tag_to_type(::Val{:ME_MatrixProduct}) = SCPCore.MatrixExpressions.MatrixProduct
ReproducibleJobs.deconstruct(A::SCPCore.MatrixExpressions.MatrixProduct{T}) where T = (A.factors,)
ReproducibleJobs.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}, (factors,)::Tuple{T}) where T =
	SCPCore.MatrixExpressions.MatrixProduct(parent(factors)) # Is this enough or do we need to convert to get the right eltype?

ReproducibleJobs.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.MatrixSum}) = true
ReproducibleJobs.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.MatrixSum}) = true
ReproducibleJobs.type_to_tag(::Type{<:SCPCore.MatrixExpressions.MatrixSum}) = TypeTag(:ME_MatrixSum)
ReproducibleJobs.tag_to_type(::Val{:ME_MatrixSum}) = SCPCore.MatrixExpressions.MatrixSum
ReproducibleJobs.deconstruct(A::SCPCore.MatrixExpressions.MatrixSum) = (A.terms,)
ReproducibleJobs.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixSum}, (terms,)::Tuple{T}) where T =
	SCPCore.MatrixExpressions.MatrixSum(parent(terms)) # Is this enough or do we need to convert to get the right eltype?

ReproducibleJobs.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.DiagGram}) = true
ReproducibleJobs.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.DiagGram}) = true
ReproducibleJobs.type_to_tag(::Type{<:SCPCore.MatrixExpressions.DiagGram}) = TypeTag(:ME_DiagGram)
ReproducibleJobs.tag_to_type(::Val{:ME_DiagGram}) = SCPCore.MatrixExpressions.DiagGram
ReproducibleJobs.deconstruct(A::SCPCore.MatrixExpressions.DiagGram{T}) where T = (A.A,)
ReproducibleJobs.reconstruct(::Type{<:SCPCore.MatrixExpressions.DiagGram}, (A,)::Tuple{T}) where T =
	SCPCore.MatrixExpressions.DiagGram(A)

ReproducibleJobs.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.Diag}) = true
ReproducibleJobs.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.Diag}) = true
ReproducibleJobs.type_to_tag(::Type{<:SCPCore.MatrixExpressions.Diag}) = TypeTag(:ME_Diag)
ReproducibleJobs.tag_to_type(::Val{:ME_Diag}) = SCPCore.MatrixExpressions.Diag
ReproducibleJobs.deconstruct(A::SCPCore.MatrixExpressions.Diag) = (A.A,)
ReproducibleJobs.reconstruct(::Type{<:SCPCore.MatrixExpressions.Diag}, (A,)::Tuple{SCPCore.MatrixProduct}) =
	SCPCore.MatrixExpressions.Diag(A)


include("types.jl")
include("projectables.jl")
include("datamatrixfunctions.jl")
include("blocks.jl")
include("internal.jl")
include("tables.jl")
include("matrix_arithmetic.jl")
include("load.jl")
include("annotate.jl")
include("filter.jl")
include("transform.jl")
include("sum_squared.jl")
include("design.jl")
include("normalize.jl")
include("nearest_neighbors.jl")
include("reduce.jl")
include("adjoint.jl")
include("transform_coords.jl")
include("pseudobulk.jl")
include("signatures.jl")
include("local_outlier_factor.jl")
include("statistical_tests.jl")
include("annotation_transfer.jl")

end
