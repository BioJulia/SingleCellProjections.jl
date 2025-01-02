# TODO: Replace filter.jl with this file.

find_matching_ids(f, df::DataFrame) =
	select(filter(f, df; view=true), 1)

find_matching_ids(::Colon, df::DataFrame) =
	select(df, 1; copycols=false) # input is considered read-only, so we don't need to copy


function _get_id_ind(df::DataFrame, ids::DataFrame)
	@assert size(ids, 2)==1
	@assert names(df, 1) == names(ids, 1)

	ind = indexin(ids[!,1], df[!,1])
	@show size(ind)
	@assert all(!isnothing, ind) # TODO: Add boolean parameter allowing that only a subset is present
	identity.(ind) # remove `Nothing` from eltype
end
_get_id_ind(df::DataFrame, ::Colon) = 1:size(df,1)

function _subset_dataframe(df, ind)
	# input is considered read-only, so only copy if we actually subset
	ind == 1:size(df,1) ? df : df[ind, :]
end

subset_annotation(df::DataFrame, ids) =
	_subset_dataframe(df, _get_id_ind(df, ids))

function subset_matrix(data::DataMatrix, var_ids, obs_ids)
	var_ind = _get_id_ind(data.var, var_ids)
	obs_ind = _get_id_ind(data.obs, obs_ids)

	var = _subset_dataframe(data.var, var_ind)
	obs = _subset_dataframe(data.obs, obs_ind)

	matrix = _subsetmatrix(data.matrix, var_ind, obs_ind)
	DataMatrix(matrix, var, obs; duplicate_var=:ignore, duplicate_obs=:ignore)
end
