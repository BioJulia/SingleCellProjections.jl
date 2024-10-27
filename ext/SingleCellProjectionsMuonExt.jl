module SingleCellProjectionsMuonExt

using SingleCellProjections
using DataFrames

if isdefined(Base, :get_extension)
	using Muon: AnnData, AlignedMapping
else
	using ..Muon: AnnData, AlignedMapping
end


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

See also: [`create_datamatrix`](@ref), [`create_obs`](@ref)
"""
SingleCellProjections.create_var(a::AnnData) =
	insertcols(a.var, 1, :id=>collect(a.var_names); makeunique=true)

"""
	create_obs(a::AnnData)

Create a `DataFrame` where the first column contains `obs` IDs and the remaining columns contain the `obs` annotations from the `AnnData` object.

See also: [`create_datamatrix`](@ref), [`create_var`](@ref)
"""
SingleCellProjections.create_obs(a::AnnData) =
	insertcols(a.obs, 1, :cell_id=>collect(a.obs_names); makeunique=true)

get_var(a::AnnData; add_var) =
	add_var ? create_var(a) : DataFrame(; id=collect(a.var_names))
get_obs(a::AnnData; add_obs) =
	add_obs ? create_obs(a) : DataFrame(; cell_id=collect(a.obs_names))


function convert_matrix(::Type{T}, X) where T
	eltype(X) <: T && return X
	convert.(T, X) # handles both sparse and dense cases, gets rid of transposes
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
  SparseMatrixCSC{Float64, Int32}
  Variables: id
  Observations: cell_id
```

See also: [`create_var`](@ref), [`create_obs`](@ref)
"""
function SingleCellProjections.create_datamatrix(::Type{T}, a::AnnData; add_var=false, add_obs=false) where T
	X = _transpose(a.X)
	var = get_var(a; add_var)
	obs = get_obs(a; add_obs)
	X = convert_matrix(T, X)
	DataMatrix(X, var, obs)
end
SingleCellProjections.create_datamatrix(a::AnnData; kwargs...) = create_datamatrix(Any, a; kwargs...)

function SingleCellProjections.create_datamatrix(::Type{T}, am::AlignedMapping, name; add_var=false, add_obs=false) where T
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
SingleCellProjections.create_datamatrix(am::AlignedMapping, name; kwargs...) = create_datamatrix(Any, am, name; kwargs...)




end
