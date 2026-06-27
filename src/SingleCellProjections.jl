module SingleCellProjections

export
	DataMatrix,
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
			SingleCellProjectionsCore,
			project,
			load_counts,
			load_csv,
			load_h5ad,
			umap,
			tsne,
			get_matrix,
			get_var,
			get_obs,
			nvar,
			nobs,
			annotate_var,
			annotate_obs,
			add_var_column,
			add_obs_column,
			var_counts_fraction,
			var_counts_sum,
			obs_counts_fraction,
			obs_counts_sum,
			subset_annotation,
			subset_var,
			subset_obs,
			subset_matrix,
			filter_annotations,
			filter_var,
			filter_obs,
			filter_matrix,
			sctransform,
			logtransform,
			tf_idf_transform,
			designmatrix,
			negative_regression_matrix,
			normalize_matrix,
			svd,
			pca,
			loadings,
			force_layout,
			transpose,
			variance,
			std,
			relative_std,
			transform_coords,
			find_optimal_coord_transform,
			pseudobulk,
			population_matrix,
			signature,
			local_outlier_factor,
			ftest,
			ttest,
			transfer_annotation,
			create_table,
			get_colnames,
			get_id_colname,
			get_value_colname,
			get_columns,
			id_column,
			value_column,
			annotation,
			column_data,
			id_column_data,
			value_column_data,
			table_nrow,
			table_ncol,
			add_column,
			table_leftjoin,
			table_hcat,
			transform_annotation
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


"""
    SCP.load_csv(filepath; kwargs...) -> Job

Load a CSV or TSV file as a table `Job`. The file path is automatically checksummed for
cache invalidation. Requires the `CSV` package to be loaded.

See also [`load_counts`](@ref).
"""
function load_csv end

"""
    SCP.load_h5ad([T], filepath; layer=nothing, obsm=nothing, obsp=nothing, varm=nothing, varp=nothing, kwargs...) -> Job

Load a .h5ad (AnnData) file as a `DataMatrix` `Job`. Requires the `Muon` package to be loaded.

The optional type parameter `T` determines the `eltype` of the matrix. If specified, the
matrix will be converted (e.g. `Int` for count matrices stored as floats).

By default, the main matrix `X` is loaded. Use one of the following mutually exclusive
kwargs to load from a different source:
* `layer` — a named layer from `layers` (e.g. `"raw_counts"`)
* `obsm` — observation embeddings (e.g. `"X_umap"`), var is set to synthetic dimension IDs
* `obsp` — observation pairwise matrix, both var and obs are set to obs annotations
* `varm` — variable embeddings, obs is set to synthetic dimension IDs
* `varp` — variable pairwise matrix, both var and obs are set to var annotations

# Examples

Load the main matrix `X`.
```julia
julia> SCP.load_h5ad("data.h5ad")
```

Load raw counts. Note that we want to specify the eltype `Int`, because h5ad typically stores counts as Float32.
```julia
julia> SCP.load_h5ad(Int, "data.h5ad"; layer="raw_counts")
```

Load a UMAP embedding.
```julia
julia> SCP.load_h5ad("data.h5ad"; obsm="X_umap")
```

See also [`load_counts`](@ref), [`load_csv`](@ref).
"""
function load_h5ad end

"""
    SCP.umap(data; ndim, seed=1234, kwargs...) -> Job

Compute a UMAP embedding of `data` with `ndim` dimensions. Returns a `DataMatrix` with
UMAP dimensions as variables. Requires the `UMAP` package to be loaded.

`seed` is used to reset the global RNG for reproducibility, but results may still vary
across runs due to threading differences in the UMAP nearest neighbor search.

Additional keyword arguments are forwarded to `UMAP.fit`.

See also [`force_layout`](@ref), [`tsne`](@ref).
"""
function umap end

"""
    SCP.tsne(data; ndim=3, kwargs...) -> Job

Compute a t-SNE embedding of `data` with `ndim` dimensions. Returns a `DataMatrix` with
t-SNE dimensions as variables. Requires the `TSne` package to be loaded.

Additional keyword arguments (`max_iter`, `perplexity`, etc.) are forwarded to `TSne.tsne`.

See also [`force_layout`](@ref), [`umap`](@ref).
"""
function tsne end


include("Impl/Impl.jl")

using Impl: DataMatrixFunction


include("projectables.jl")
include("datamatrixfunctions.jl")
include("internal.jl")
include("tables.jl")
include("load.jl")
include("annotate.jl")
include("filter.jl")
include("transform.jl")
include("sum_squared.jl")
include("design.jl")
include("normalize.jl")
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
