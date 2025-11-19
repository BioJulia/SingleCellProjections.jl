function subset_matrix(::Preprocessing, data; var_ids=nothing, obs_ids=nothing)
	var_ind = var_ids === nothing ? Colon() : indexin_spec(var_ids, id_column_spec(get_var_spec(data)); not_found=:error)
	obs_ind = obs_ids === nothing ? Colon() : indexin_spec(obs_ids, id_column_spec(get_obs_spec(data)); not_found=:error)
	create_datamatrix_getindex_spec(data; var_ind, obs_ind)
end


Jobs.subset_matrix(data, var_ids, obs_ids) =
	Job(create_spec(Preprocess(subset_matrix), data; var_ids, obs_ids))
Jobs.subset_var(data, var_ids) =
	Job(create_spec(Preprocess(subset_matrix), data; var_ids))
Jobs.subset_obs(data, obs_ids) =
	Job(create_spec(Preprocess(subset_matrix), data; obs_ids))



function filter_matrix(::Preprocessing, data; fvar=nothing, fobs=nothing, project_var_ids=nothing, project_obs_ids=nothing)
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

	var_ind = create_find_matching_ind_spec(fvar, get_var_spec(data); project_ids=project_var_ids)
	obs_ind = create_find_matching_ind_spec(fobs, get_obs_spec(data); project_ids=project_obs_ids)

	create_datamatrix_getindex_spec(data; var_ind, obs_ind)
end


Jobs.filter_matrix(fvar, fobs, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fvar, fobs))
Jobs.filter_var(fvar, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fvar))
Jobs.filter_obs(fobs, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fobs))
