function _subset_ind(f::Union{Var,Obs}, data; kwargs...)
	ids = get(kwargs, f isa Var ? :var_ids : :obs_ids, nothing)
	ids === nothing && return nothing
	s = get_spec(f, data)
	create_ids_to_indices_spec(s, ids)
end

function subset_matrix(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	var_ind = _subset_ind(Var(), data; kwargs...)
	obs_ind = _subset_ind(Obs(), data; kwargs...)
	if var_ind !== nothing && obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind), obs_ind=prefetch(obs_ind))
	elseif var_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind))
	elseif obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; obs_ind=prefetch(obs_ind))
	else
		matrix_spec
	end
end
function subset_matrix(f::Union{Var,Obs}, data; kwargs...)
	s = get_spec(f, data)
	ind = _subset_ind(f, data; kwargs...)
	ind === nothing && return s
	create_annotation_getindex_spec(s, prefetch(ind))
end


Jobs.subset_matrix(data, var_ids, obs_ids; kwargs...) =
	Job(create_spec(DataMatrixFunc(subset_matrix), data; kwargs..., var_ids, obs_ids))
Jobs.subset_var(data, var_ids; kwargs...) =
	Job(create_spec(DataMatrixFunc(subset_matrix), data; kwargs..., var_ids))
Jobs.subset_obs(data, obs_ids; kwargs...) =
	Job(create_spec(DataMatrixFunc(subset_matrix), data; kwargs..., obs_ids))





function _filter_ind(f::Union{Var,Obs}, data; kwargs...)
	fun = get(kwargs, f isa Var ? :fvar : :fobs, nothing)
	fun === nothing && return nothing
	s = get_spec(f, data)
	project_ids = kwargs[f isa Var ? :project_var_ids : :project_obs_ids]
	id_spec = create_find_matching_ids_spec(fun, s; project_ids)
	create_ids_to_indices_spec(s, id_spec)
end

function filter_matrix(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	var_ind = _filter_ind(Var(), data; kwargs...)
	obs_ind = _filter_ind(Obs(), data; kwargs...)
	if var_ind !== nothing && obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind), obs_ind=prefetch(obs_ind))
	elseif var_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind))
	elseif obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; obs_ind=prefetch(obs_ind))
	else
		matrix_spec
	end
end
function filter_matrix(f::Union{Var,Obs}, data; kwargs...)
	s = get_spec(f, data)
	ind = _filter_ind(f, data; kwargs...)
	ind === nothing && return s
	create_annotation_getindex_spec(s, prefetch(ind))
end



# Can we share more code with subset_ functions above?
Jobs.filter_matrix(fvar, fobs, data; project_var_ids=:intersect, project_obs_ids=:no, kwargs...) =
	Job(create_spec(DataMatrixFunc(filter_matrix), data;  kwargs..., fvar, fobs, project_var_ids, project_obs_ids))
Jobs.filter_var(fvar, data; project_ids=:intersect, kwargs...) =
	Job(create_spec(DataMatrixFunc(filter_matrix), data;  kwargs..., fvar, project_var_ids=project_ids))
Jobs.filter_obs(fobs, data; project_ids=:no, kwargs...) =
	Job(create_spec(DataMatrixFunc(filter_matrix), data;  kwargs..., fobs, project_obs_ids=project_ids))
