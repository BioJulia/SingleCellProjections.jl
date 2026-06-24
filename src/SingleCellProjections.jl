module SingleCellProjections

export
	DataMatrix,
	Jobs, # TODO: remove probably
	categorical_covariate,
	numerical_covariate,
	twogroup_covariate,
	rot2d,
	flipx2d,
	flipy2d,
	rotx,
	roty,
	rotz,
	flipx3d,
	flipy3d,
	flipz3d

# Use public keyword in Julia versions where it is available
if VERSION >= v"1.11.0-DEV.469"
	let str = """
		public
			SingleCellProjectionsCore
		"""
		eval(Meta.parse(str))
	end
end

include("SingleCellProjectionsCore/SingleCellProjectionsCore.jl")

import .SingleCellProjectionsCore as SCPCore
using .SCPCore: DataMatrix, Blocks

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



module Jobs
	function project end
	function load_counts end
	function get_matrix end
	function get_var end
	function get_obs end
	function nvar end
	function nobs end
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
	function designmatrix end
	function negative_regression_matrix end # should this be public?
	function normalize_matrix end
	function svd end
	function pca end
	function loadings end
	function force_layout end
	function transpose end
	function variance end
	function std end
	function relative_std end
	function transform_coords end
	function find_optimal_coord_transform end
	function pseudobulk end
	function population_matrix end
	function signature end
	function local_outlier_factor end
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

	"""
	    Jobs.load_csv(filepath; kwargs...) -> Job

	Load a CSV or TSV file as a table `Job`. The file path is automatically checksummed for
	cache invalidation. Requires the `CSV` package to be loaded.

	See also [`Jobs.load_counts`](@ref).
	"""
	function load_csv end

	"""
	    Jobs.umap(data; ndim, seed=1234, kwargs...) -> Job

	Compute a UMAP embedding of `data` with `ndim` dimensions. Returns a `DataMatrix` with
	UMAP dimensions as variables. Requires the `UMAP` package to be loaded.

	`seed` is used to reset the global RNG for reproducibility, but results may still vary
	across runs due to threading differences in the UMAP nearest neighbor search.

	Additional keyword arguments are forwarded to `UMAP.fit`.

	See also [`Jobs.force_layout`](@ref), [`Jobs.tsne`](@ref).
	"""
	function umap end

	"""
	    Jobs.tsne(data; ndim=3, kwargs...) -> Job

	Compute a t-SNE embedding of `data` with `ndim` dimensions. Returns a `DataMatrix` with
	t-SNE dimensions as variables. Requires the `TSne` package to be loaded.

	Additional keyword arguments (`max_iter`, `perplexity`, etc.) are forwarded to `TSne.tsne`.

	See also [`Jobs.force_layout`](@ref), [`Jobs.umap`](@ref).
	"""
	function tsne end
end



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


# include("precompile.jl")

function register_scp_functions!(scheduler::ReproducibleJobs.Scheduler)
	ReproducibleJobs.register_function!(scheduler, mean)
	ReproducibleJobs.register_function!(scheduler, /)
end
register_scp_functions!() = register_scp_functions!(ReproducibleJobs.get_scheduler())


function __init__()
	register_scp_functions!()
end


end
