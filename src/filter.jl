function subset_matrix(data; var_ids=:, obs_ids=:)
	var_ind = create_ids_to_indices_spec(get_var_spec(data), var_ids)
	obs_ind = create_ids_to_indices_spec(get_obs_spec(data), obs_ids)
	create_datamatrix_getindex_spec(data; var_ind, obs_ind)
end


Jobs.subset_matrix(data, var_ids, obs_ids; kwargs...) =
	Job(create_spec(Preprocess(subset_matrix), data; kwargs..., var_ids, obs_ids))
Jobs.subset_var(data, var_ids; kwargs...) =
	Job(create_spec(Preprocess(subset_matrix), data; kwargs..., var_ids))
Jobs.subset_obs(data, obs_ids; kwargs...) =
	Job(create_spec(Preprocess(subset_matrix), data; kwargs..., obs_ids))




function _filter_ind(f, s; project_ids)
	id_spec = create_find_matching_ids_spec(f, s; project_ids)
	create_ids_to_indices_spec(id_column_spec(s), id_spec)
end
function filter_matrix(data; fvar=nothing, fobs=nothing, project_var_ids=nothing, project_obs_ids=nothing)
	if fvar === nothing
		@assert project_var_ids === nothing
		fvar = Colon()
		project_var_ids = :no
	end
	if fobs === nothing
		@assert project_obs_ids === nothing
		fobs = Colon()
		project_obs_ids = :no
	end
	project_var_ids = @something(project_var_ids, :intersect)
	project_obs_ids = @something(project_obs_ids, :no)

	var_ind = _filter_ind(fvar, get_var_spec(data); project_ids=project_var_ids)
	obs_ind = _filter_ind(fobs, get_obs_spec(data); project_ids=project_obs_ids)
	create_datamatrix_getindex_spec(data; var_ind, obs_ind)
end


Jobs.filter_matrix(fvar, fobs, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fvar, fobs))
Jobs.filter_var(fvar, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fvar))
Jobs.filter_obs(fobs, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fobs))
