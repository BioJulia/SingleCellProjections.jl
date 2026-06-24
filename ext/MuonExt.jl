module MuonExt

using ReproducibleJobs
using ReproducibleJobs: create_job, cached, prefetched, ChecksummedFilePath
using SingleCellProjections
using SingleCellProjections: DataMatrixFunction, Mat, Var, Obs, table_to_compound_result, table_from_compound_result, checksummedfilepath_job, prefixed_ids_job, compute_size_job
using .SingleCellProjections.SingleCellProjectionsCore
using DataFrames
using SparseArrays: SparseMatrixCSC
import LinearAlgebra

using HDF5: h5open
using Muon: AnnData, AlignedMapping, readh5ad

function aligned_mapping_type(am::AlignedMapping)
	ref = am.ref
	am === ref.layers && return :layers
	am === ref.obsm && return :obsm
	am === ref.obsp && return :obsp
	am === ref.varm && return :varm
	am === ref.varp && return :varp
	throw(ArgumentError("Unknown AlignedMapping"))
end

"""
    create_var(a::AnnData)

Create a `DataFrame` where the first column contains `var` IDs and the remaining columns contain the `var` annotations from the `AnnData` object.

!!! note
	The interface for loading data from .h5ad files is still considered experimental and might change in a non-breaking release.

See also: [`create_datamatrix`](@ref), [`create_obs`](@ref)
"""
SingleCellProjectionsCore.create_var(a::AnnData) =
	insertcols(a.var, 1, :id=>collect(a.var_names); makeunique=true)

"""
	create_obs(a::AnnData)

Create a `DataFrame` where the first column contains `obs` IDs and the remaining columns contain the `obs` annotations from the `AnnData` object.

!!! note
	The interface for loading data from .h5ad files is still considered experimental and might change in a non-breaking release.

See also: [`create_datamatrix`](@ref), [`create_var`](@ref)
"""
SingleCellProjectionsCore.create_obs(a::AnnData) =
	insertcols(a.obs, 1, :cell_id=>collect(a.obs_names); makeunique=true)

get_var(a::AnnData; add_var) =
	add_var ? create_var(a) : DataFrame(; id=collect(a.var_names))
get_obs(a::AnnData; add_obs) =
	add_obs ? create_obs(a) : DataFrame(; cell_id=collect(a.obs_names))


function convert_matrix(::Type{T}, X) where T
	if !(eltype(X) <: T)
		convert.(T, X) # handles both sparse and dense cases, gets rid of transposes
	elseif X isa LinearAlgebra.Adjoint
		copy(X) # materialize transpose
	else
		X
	end
end



function _transpose(X::PermutedDimsArray)
	Xt = parent(X)
	@assert PermutedDimsArray(Xt, (2,1)) === X
	Xt
end
_transpose(X) = X'


"""
	create_datamatrix([T], a::AnnData; add_var=false, add_obs=false)
	create_datamatrix([T], am::AlignedMapping, name; add_var=false, add_obs=false)

Creates a `DataMatrix` from an `AnnData` object.
By default, the main matrix `X` is retrieved from `a::AnnData`.
It is also possible to create `DataMatrices` from named objects in: `a.layers`, `a.obsm`, `a.obsp`, `a.varm` and `a.varp`. See examples below.

The optional parameter `T` determines the `eltype` of the returned matrix. If specified, the matrix will be converted to have this `eltype`.

kwargs:
* add_var: Add `var` from the AnnData object to the returned `DataMatrix` (when applicable).
* add_obs: Add `obs` from the AnnData object to the returned `DataMatrix` (when applicable).

!!! note
	The interface for loading data from .h5ad files is still considered experimental and might change in a non-breaking release.

# Examples

All examples below assume that an AnnData object has been loaded first:
```julia
julia> using Muon

julia> a = readh5ad("path/to/file.h5ad");
```

* Load the main matrix `X` from an AnnData object.
```julia
julia> create_datamatrix(a)
DataMatrix (123 variables and 456 observations)
  SparseMatrixCSC{Float32, Int32}
  Variables: id
  Observations: cell_id
```

* Load the main matrix `X` from an AnnData object, and add `var`/`obs` annotations.
```julia
julia> create_datamatrix(a; add_var=true, add_obs=true)
DataMatrix (123 variables and 456 observations)
  SparseMatrixCSC{Float32, Int32}
  Variables: id, feature_type, ...
  Observations: cell_id, cell_type, ...
```

* Load the main matrix `X` from an AnnData object, with eltype `Int`. NB: This will fail if the matrix is not a count matrix.
```julia
julia> create_datamatrix(Int, a)
DataMatrix (123 variables and 456 observations)
  SparseMatrixCSC{Int64, Int32}
  Variables: id
  Observations: cell_id
```

* Load the matrix named `raw_counts` from `layers`, with eltype `Int`. NB: This will fail if the matrix is not a count matrix.
```julia
julia> create_datamatrix(Int, a.layers, "raw_counts")
DataMatrix (123 variables and 456 observations)
  SparseMatrixCSC{Int64, Int32}
  Variables: id
  Observations: cell_id
```

* Load the matrix named `UMAP` from `obsm`.
```julia
julia> create_datamatrix(a.obsm, "UMAP")
DataMatrix (2 variables and 456 observations)
  Matrix{Float64}
  Variables: id
  Observations: cell_id
```

See also: [`create_var`](@ref), [`create_obs`](@ref)
"""
function SingleCellProjectionsCore.create_datamatrix(::Type{T}, a::AnnData; add_var=false, add_obs=false) where T
	X = _transpose(a.X)
	var = get_var(a; add_var)
	obs = get_obs(a; add_obs)
	X = convert_matrix(T, X)
	DataMatrix(X, var, obs)
end
SingleCellProjectionsCore.create_datamatrix(a::AnnData; kwargs...) = create_datamatrix(Any, a; kwargs...)

function SingleCellProjectionsCore.create_datamatrix(::Type{T}, am::AlignedMapping, name; add_var=false, add_obs=false) where T
	a = am.ref
	am_type = aligned_mapping_type(am)
	X = am[name]

	new_ids = nothing
	if X isa DataFrame
		new_ids = names(X)
		X = Matrix(X)
	end

	@assert ndims(X) == 2 "Expected DataMatrix to have 2 dimensions, got $(ndims(X))"

	if am_type == :layers
		X = _transpose(X)
		var = get_var(a; add_var)
		obs = get_obs(a; add_obs)
	elseif am_type == :obsm
		X = _transpose(X)
		id = @something new_ids string.("Dim", 1:size(X,1))
		var = DataFrame(; id)
		obs = get_obs(a; add_obs)
	elseif am_type == :obsp
		X = _transpose(X)
		var = obs = get_obs(a; add_obs)
	elseif am_type == :varm
		var = get_var(a; add_var)
		id = @something new_ids string.("Dim", 1:size(X,2))
		obs = DataFrame(; id)
	elseif am_type == :varp
		var = obs = get_var(a; add_var)
	end

	X = convert_matrix(T, X)
	DataMatrix(X, var, obs)
end
SingleCellProjectionsCore.create_datamatrix(am::AlignedMapping, name; kwargs...) = create_datamatrix(Any, am, name; kwargs...)



# --- Jobs-based h5ad loading ---

function _read_h5ad(f, filepath)
	@assert filepath isa ChecksummedFilePath
	# This workaround is needed because Muon by default opens backed files in "r+" which can change the mtime.
	h5open(string(filepath), "r") do fid
		ann = AnnData(fid, true, false)
		f(ann)
	end
end


function load_h5ad_var_impl(filepath)
	_read_h5ad(filepath) do ann
		df = insertcols(ann.var, 1, :id => collect(ann.var_names); makeunique=true)
		table_to_compound_result(df)
	end
end
load_h5ad_var_job(filepath) = create_job(load_h5ad_var_impl, filepath; __version=v"0.1.0")

function load_h5ad_obs_impl(filepath)
	_read_h5ad(filepath) do ann
		df = insertcols(ann.obs, 1, :cell_id => collect(ann.obs_names); makeunique=true)
		table_to_compound_result(df)
	end
end
load_h5ad_obs_job(filepath) = create_job(load_h5ad_obs_impl, filepath; __version=v"0.1.0")

function load_h5ad_matrix_impl(filepath; T, layer=nothing, obsm=nothing, obsp=nothing, varm=nothing, varp=nothing, row_block_size=1024, col_block_size=1024)
	_read_h5ad(filepath) do ann
		# X and layers are lazy (backed) and need read(), obsm/obsp/varm/varp are eagerly loaded
		X = if layer !== nothing
			read(ann.layers[layer])
		elseif obsm !== nothing
			ann.obsm[obsm]
		elseif obsp !== nothing
			ann.obsp[obsp]
		elseif varm !== nothing
			ann.varm[varm]
		elseif varp !== nothing
			ann.varp[varp]
		else
			read(ann.X)
		end
		if varm === nothing && varp === nothing
			X = _transpose(X)
		end
		X = convert_matrix(T, X)
		if X isa SparseMatrixCSC
			X = SingleCellProjectionsCore.blockify(X; row_block_size, col_block_size)
		end
		X
	end
end
load_h5ad_matrix_job(filepath; kwargs...) = create_job(load_h5ad_matrix_impl, filepath; T=Any, kwargs..., __version=v"0.1.0")


load_h5ad(::Mat, filepath; kwargs...) = load_h5ad_matrix_job(filepath; kwargs...)

function load_h5ad(::Var, filepath; obsm=nothing, obsp=nothing, kwargs...)
	if obsm !== nothing
		mat_job = load_h5ad(Mat(), filepath; obsm, kwargs...)
		prefixed_ids_job("id", "Dim", prefetched(compute_size_job(mat_job, 1)))
	elseif obsp !== nothing
		table_from_compound_result(cached(load_h5ad_obs_job(filepath)))
	else
		table_from_compound_result(cached(load_h5ad_var_job(filepath)))
	end
end

function load_h5ad(::Obs, filepath; varm=nothing, varp=nothing, kwargs...)
	if varm !== nothing
		mat_job = load_h5ad(Mat(), filepath; varm, kwargs...)
		prefixed_ids_job("id", "Dim", prefetched(compute_size_job(mat_job, 2)))
	elseif varp !== nothing
		table_from_compound_result(cached(load_h5ad_var_job(filepath)))
	else
		table_from_compound_result(cached(load_h5ad_obs_job(filepath)))
	end
end

function Jobs.load_h5ad(filepath; kwargs...)
	if count(key->haskey(kwargs,key), (:layer, :obsm, :obsp, :varm, :varp)) > 1
		throw(ArgumentError("At most one of layer, obsm, obsp, varm, varp can be specified."))
	end

	filepath_job = checksummedfilepath_job(filepath)
	create_job(DataMatrixFunction(load_h5ad), filepath_job; kwargs...)
end
Jobs.load_h5ad(::Type{T}, filepath; kwargs...) where T = Jobs.load_h5ad(filepath; T, kwargs...)


end
