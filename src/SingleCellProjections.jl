module SingleCellProjections

export
	SingleCellProjectionsCore, # TODO: make public instead
	DataMatrix,
	Jobs, # TODO: remove probably
	categorical_covariate,
	numerical_covariate,
	twogroup_covariate,
	rot2d, # experimental
	flipx2d, # experimental
	flipy2d, # experimental
	rotx, # experimental
	roty, # experimental
	rotz, # experimental
	flipx3d, # experimental
	flipy3d, # experimental
	flipz3d # experimental

include("SingleCellProjectionsCore/SingleCellProjectionsCore.jl")

import .SingleCellProjectionsCore as SCPCore
using .SCPCore: DataMatrix, categorical_covariate, numerical_covariate, twogroup_covariate

import SCTransform
import SingleCell10x
using DataFrames
import StableHashTraits
import LinearAlgebra
using SparseArrays: sparse
import StatsBase

using ChunkSplitters
using OhMyThreads

using ReproducibleJobs
using ReproducibleJobs: Deduplicators, create_spec, cached, SpecArgs, ChecksummedFilePath, Preprocessing, checksummedfilepath_spec, ifelse_spec
using .Deduplicators: ROArray, ROVec, ROMat, ROBitArray, ROBitVec, ROBitMat, TypeTag

using ReadOnlyArrays: ReadOnlyVector

using StyledStrings # For Spec printing

# ReproducibleJobs.unmanage_rec(x::DataMatrix) =
# 	DataMatrix(ReproducibleJobs.unmanage_rec.((x.matrix, x.var, x.obs))...)
# ReproducibleJobs.unmanage_rec(x::SCPCore.AbstractValueVector) = x
# ReproducibleJobs.unmanage_rec(x::SCPCore.AbstractValueVectorModel) = x
# ReproducibleJobs.unmanage_rec(x::SCPCore.ProjectionModel) = x


# TODO: Where to put these?
Deduplicators.deduplicate_type(::Type{<:DataMatrix}) = true
Deduplicators.deconstruct_type(::Type{<:DataMatrix}) = true
Deduplicators.type_to_tag(::Type{<:DataMatrix}) = TypeTag(:DataMatrix)
Deduplicators.tag_to_type(::Val{:DataMatrix}) = DataMatrix
Deduplicators.deconstruct(data::DataMatrix{T,Tv,To}) where {T,Tv,To} = (data.matrix, data.var, data.obs)
Deduplicators.reconstruct(::Type{<:DataMatrix}, (matrix,var,obs)::Tuple{T,Tv,To}) where {T,Tv,To} =
	DataMatrix(matrix, var, obs; duplicate_var=:ignore, duplicate_obs=:ignore) # Avoid doing validation when reconstructing!


# TODO: Where to put these?
Deduplicators.deduplicate_type(::Type{<:SCPCore.AbstractCovariateDesc}) = false
Deduplicators.deconstruct_weak_rec(x::T) where T<:SCPCore.AbstractCovariateDesc = x
Deduplicators.reconstruct_weak_rec(x::T) where T<:SCPCore.AbstractCovariateDesc = x

Deduplicators.deduplicate_type(::Type{<:SCPCore.AbstractValueVectorModel}) = false
Deduplicators.deconstruct_weak_rec(x::T) where T<:SCPCore.AbstractValueVectorModel = x
Deduplicators.reconstruct_weak_rec(x::T) where T<:SCPCore.AbstractValueVectorModel = x

Deduplicators.deduplicate_type(::Type{<:SCPCore.CategoricalValueVectorModel}) = true
Deduplicators.deconstruct_type(::Type{<:SCPCore.CategoricalValueVectorModel}) = true
Deduplicators.type_to_tag(::Type{<:SCPCore.CategoricalValueVectorModel}) = TypeTag(:CategoricalValueVectorModel)
Deduplicators.tag_to_type(::Val{:CategoricalValueVectorModel}) = SCPCore.CategoricalValueVectorModel
Deduplicators.deconstruct(m::SCPCore.CategoricalValueVectorModel{T}) where T = (m.categories,)
Deduplicators.reconstruct(::Type{<:SCPCore.CategoricalValueVectorModel}, (categories,)::Tuple{T}) where T =
	SCPCore.CategoricalValueVectorModel(parent(categories)) # Refactoring TODO: Avoid doing validation when reconstructing!



# TODO: Where to put these?
Deduplicators.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.MatrixRef}) = true
Deduplicators.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.MatrixRef}) = true
Deduplicators.type_to_tag(::Type{<:SCPCore.MatrixExpressions.MatrixRef}) = TypeTag(:ME_MatrixRef)
Deduplicators.tag_to_type(::Val{:ME_MatrixRef}) = SCPCore.MatrixExpressions.MatrixRef
Deduplicators.deconstruct(A::SCPCore.MatrixExpressions.MatrixRef{T}) where T = (A.name, A.matrix)
Deduplicators.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixRef}, (name,matrix)::Tuple{Symbol,T}) where T =
	SCPCore.MatrixExpressions.MatrixRef(name, matrix)
Deduplicators.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixRef}, (name,matrix)::Tuple{Symbol,ROArray{T}}) where T =
	SCPCore.MatrixExpressions.MatrixRef(name, parent(matrix))

Deduplicators.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}) = true
Deduplicators.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}) = true
Deduplicators.type_to_tag(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}) = TypeTag(:ME_MatrixProduct)
Deduplicators.tag_to_type(::Val{:ME_MatrixProduct}) = SCPCore.MatrixExpressions.MatrixProduct
Deduplicators.deconstruct(A::SCPCore.MatrixExpressions.MatrixProduct{T}) where T = (A.factors,)
Deduplicators.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixProduct}, (factors,)::Tuple{T}) where T =
	SCPCore.MatrixExpressions.MatrixProduct(parent(factors)) # Is this enough or do we need to convert to get the right eltype?

Deduplicators.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.MatrixSum}) = true
Deduplicators.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.MatrixSum}) = true
Deduplicators.type_to_tag(::Type{<:SCPCore.MatrixExpressions.MatrixSum}) = TypeTag(:ME_MatrixSum)
Deduplicators.tag_to_type(::Val{:ME_MatrixSum}) = SCPCore.MatrixExpressions.MatrixSum
Deduplicators.deconstruct(A::SCPCore.MatrixExpressions.MatrixSum) = (A.terms,)
Deduplicators.reconstruct(::Type{<:SCPCore.MatrixExpressions.MatrixSum}, (terms,)::Tuple{T}) where T =
	SCPCore.MatrixExpressions.MatrixSum(parent(terms)) # Is this enough or do we need to convert to get the right eltype?

Deduplicators.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.DiagGram}) = true
Deduplicators.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.DiagGram}) = true
Deduplicators.type_to_tag(::Type{<:SCPCore.MatrixExpressions.DiagGram}) = TypeTag(:ME_DiagGram)
Deduplicators.tag_to_type(::Val{:ME_DiagGram}) = SCPCore.MatrixExpressions.DiagGram
Deduplicators.deconstruct(A::SCPCore.MatrixExpressions.DiagGram{T}) where T = (A.A,)
Deduplicators.reconstruct(::Type{<:SCPCore.MatrixExpressions.DiagGram}, (A,)::Tuple{T}) where T =
	SCPCore.MatrixExpressions.DiagGram(A)

Deduplicators.deduplicate_type(::Type{<:SCPCore.MatrixExpressions.Diag}) = true
Deduplicators.deconstruct_type(::Type{<:SCPCore.MatrixExpressions.Diag}) = true
Deduplicators.type_to_tag(::Type{<:SCPCore.MatrixExpressions.Diag}) = TypeTag(:ME_Diag)
Deduplicators.tag_to_type(::Val{:ME_Diag}) = SCPCore.MatrixExpressions.Diag
Deduplicators.deconstruct(A::SCPCore.MatrixExpressions.Diag) = (A.A,)
Deduplicators.reconstruct(::Type{<:SCPCore.MatrixExpressions.Diag}, (A,)::Tuple{SCPCore.MatrixProduct}) =
	SCPCore.MatrixExpressions.Diag(A)



# TODO: This is a temporary solution when refactoring, remove
module Jobs
	function project end
	function load_counts end
	function get_matrix end
	function get_var end
	function get_obs end
	function nvar end
	function nobs end
	function annotate end
	function annotate_var end
	function annotate_obs end
	function add_var_column end
	function add_obs_column end
	function var_counts_fraction end
	function var_counts_sum end
	function obs_counts_fraction end
	function obs_counts_sum end
	function subset_annotation end
	function subset_var end
	function subset_obs end
	function subset_matrix end
	function filter_annotations end
	function filter_var end
	function filter_obs end
	function filter_matrix end
	function sctransform end
	function logtransform end
	function tf_idf_transform end
	# function center_matrix end # TEMP
	function designmatrix end
	function negative_regression_matrix end
	function normalize_matrix end
	function svd end
	function pca end
	function loadings end
	function force_layout end
	function transpose end

	function transform_coords end

	function pseudobulk end
	function population_matrix end

	function ftest end
	function ttest end

	function transfer_annotation end

	function create_table end
	function get_colnames end
	function get_id_colname end
	function get_value_colname end
	function get_columns end
	function id_column end
	function value_column end
	function annotation end
	function column_data end
	function id_column_data end
	function value_column_data end
	function table_nrow end
	function table_ncol end
	function add_column end
	function table_leftjoin end
	function table_hcat end
	function transform_annotation end

	function load_csv end
	function umap end
	function tsne end
end



include("types.jl")
include("projectables.jl")
include("datamatrixfunctions.jl")
include("internal.jl")
include("tables.jl")
include("matrix_arithmetic.jl")
include("load.jl")
include("annotate.jl")
include("filter.jl")
include("transform.jl")
include("design.jl")
include("normalize.jl")
include("nearest_neighbors.jl")
include("reduce.jl")
include("adjoint.jl")
include("transform_coords.jl")
include("pseudobulk.jl")
include("statistical_tests.jl")
include("annotation_transfer.jl")


# include("precompile.jl")


end
