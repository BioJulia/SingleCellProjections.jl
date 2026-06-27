function subset_matrix(::Preprocessing, data; var_ids=nothing, obs_ids=nothing)
	var_ind_args = var_ids === nothing ? (;) : (; var_ind=indexin_job(var_ids, id_column_job(get_var_job(data)); not_found=:error))
	obs_ind_args = obs_ids === nothing ? (;) : (; obs_ind=indexin_job(obs_ids, id_column_job(get_obs_job(data)); not_found=:error))
	create_datamatrix_getindex_job(data; var_ind_args..., obs_ind_args...)
end


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

	var_ind = create_find_matching_ind_job(fvar, get_var_job(data); project_ids=project_var_ids)
	obs_ind = create_find_matching_ind_job(fobs, get_obs_job(data); project_ids=project_obs_ids)

	create_datamatrix_getindex_job(data; var_ind, obs_ind)
end
