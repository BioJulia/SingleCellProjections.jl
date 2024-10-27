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

create_var(a::AnnData) =
	insertcols(a.var, 1, :id=>collect(a.var_names); makeunique=true)
create_obs(a::AnnData) =
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
