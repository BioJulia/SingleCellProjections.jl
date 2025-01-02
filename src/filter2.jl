# TODO: Replace filter.jl with this file.

find_matching_ids(f, df::DataFrame) =
	select(filter(f, df; view=true), 1)

find_matching_ids(::Colon, df::DataFrame) =
	select(df, 1; copycols=false) # input is considered read-only, so we don't need to copy

# find_matching_ids(f) = df->find_matching_ids(f, df)


function subset_annotation(df::DataFrame, ids::DataFrame)
	@assert size(ids, 2)==1
	@assert names(df, 1) == names(ids, 1)

	ind = indexin(df[!,1], ids[!,1])
	@assert all(!isnothing, ind) # TODO: Add boolean parameter allowing that only a subset is present
	ind = identity.(ind) # remove Nothing from eltype
	df[ind,:]
end

function subset_datamatrix(data::DataMatrix, var_ids, obs_ids)
	@assert size(var_ids, 2)==1
	@assert size(obs_ids, 2)==1
	@assert names(data.var, 1) == names(var_ids, 1)
	@assert names(data.obs, 1) == names(obs_ids, 1)

	var_ind = indexin(data.var[!,1], ids[!,1])
	@assert all(!isnothing, var_ind) # TODO: Add boolean parameter allowing that only a subset is present
	var_ind = identity.(var_ind) # remove Nothing from eltype

	obs_ind = indexin(data.obs[!,1], ids[!,1])
	@assert all(!isnothing, obs_ind) # TODO: Add boolean parameter allowing that only a subset is present
	obs_ind = identity.(obs_ind) # remove Nothing from eltype

	# TODO: Consider returning data.obs/var directly if var_ind/obs_ind == 1:N (instead of copying),
	#       since inputs are considered read-only
	matrix = _subsetmatrix(data.matrix, var_ind, obs_ind)
	var = data.var[var_ind,:]
	obs = data.obs[obs_ind,:]
	DataMatrix(matrix, var, obs; duplicate_var=:ignore, duplicate_obs=:ignore)
end
