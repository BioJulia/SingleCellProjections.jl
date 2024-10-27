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


function get_var(a::AnnData; add_var)
	id = collect(a.var_names)
	if add_var
		var = a.var
		var = insertcols(var, 1, :id=>id; makeunique=true)
		return var
	else
		return DataFrame(; id)
	end
end
function get_obs(a::AnnData; add_obs)
	cell_id = collect(a.obs_names)
	if add_obs
		obs = a.obs
		obs = insertcols(obs, 1, :cell_id=>cell_id; makeunique=true)
		return obs
	else
		return DataFrame(; cell_id)
	end
end



function SingleCellProjections.create_datamatrix(a::AnnData; add_var=false, add_obs=false)
	X = a.X'
	var = get_var(a; add_var)
	obs = get_obs(a; add_obs)
	DataMatrix(X, var, obs)
end


function SingleCellProjections.create_datamatrix(am::AlignedMapping, name; add_var=false, add_obs=false)
	a = am.ref
	am_type = aligned_mapping_type(am)

	X = am[name]
	if am_type in (:layers, :obsm, :obsp)
	end

	if am_type == :layers
		X = X'
		var = get_var(a; add_var)
		obs = get_obs(a; add_obs)
	elseif am_type == :obsm
		X = X'
		var = string.("Dim", 1:size(X,1))
		obs = get_obs(a; add_obs)
	elseif am_type == :obsp
		X = X'
		var = obs = get_obs(a; add_obs)
	elseif am_type == :varm
		dim1 = get_var(a; add_var)
		dim2 = string.("Dim", 1:size(X,2))
	elseif am_type == :varp
		var = obs = get_var(a; add_var)
	end

	DataMatrix(X, var, obs)
end



end
