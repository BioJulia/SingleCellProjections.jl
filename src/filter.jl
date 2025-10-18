# # TODO: Use `:` instead of `nothing` and get rid of it at a preprocessing step after projecting
# function _subset_ind(f::Union{Var,Obs}, data; kwargs...)
# 	ids = get(kwargs, f isa Var ? :var_ids : :obs_ids, nothing)
# 	ids === nothing && return nothing
# 	s = get_spec(f, data)
# 	create_ids_to_indices_spec(s, ids)
# end

# # TODO: Use `:` instead of `nothing` and get rid of it at a preprocessing step after projecting
# function subset_matrix(::Mat, data; var_ids=:, obs_ids=:)
# 	matrix_spec = get_matrix_spec(data)
# 	# var_ind = _subset_ind(Var(), data; kwargs...)
# 	# obs_ind = _subset_ind(Obs(), data; kwargs...)
# 	var_ind = create_ids_to_indices_spec(get_var_spec(data), var_ids)
# 	obs_ind = create_ids_to_indices_spec(get_obs_spec(data), obs_ids)

# 	nvar = datamatrix_nvar_spec(data)
# 	nobs = datamatrix_nobs_spec(data)

# 	create_matrix_getindex_spec(matrix_spec; var_ind=prefetched(var_ind), obs_ind=prefetched(obs_ind), nvar, nobs)

# 	# if var_ind !== nothing && obs_ind !== nothing
# 	# 	create_matrix_getindex_spec(matrix_spec; var_ind=prefetched(var_ind), obs_ind=prefetched(obs_ind))
# 	# elseif var_ind !== nothing
# 	# 	create_matrix_getindex_spec(matrix_spec; var_ind=prefetched(var_ind))
# 	# elseif obs_ind !== nothing
# 	# 	create_matrix_getindex_spec(matrix_spec; obs_ind=prefetched(obs_ind))
# 	# else
# 	# 	matrix_spec
# 	# end
# end
# # function subset_matrix(f::Union{Var,Obs}, data; kwargs...)
# # 	s = get_spec(f, data)
# # 	ind = _subset_ind(f, data; kwargs...)
# # 	ind === nothing && return s
# # 	create_annotation_getindex_spec(s, prefetched(ind))
# # end
# # function subset_matrix(::Var, data; var_ids=:, kwargs...)
# # 	var_spec = get_var_spec(data)
# # 	var_ind = create_ids_to_indices_spec(var_spec, var_ids)
# # 	create_annotation_getindex_spec(var_spec, var_ind)
# # end
# function subset_matrix(f::Union{Var,Obs}, data; var_ids=:, obs_ids=:)
# 	ids = f === Var() ? var_ids : obs_ids
# 	s = get_spec(f, data)
# 	ind = create_ids_to_indices_spec(s, ids)
# 	create_annotation_getindex_spec(s, ind)
# end



# Jobs.subset_matrix(data, var_ids, obs_ids; kwargs...) =
# 	Job(create_spec(DataMatrixFunction(subset_matrix), data; kwargs..., var_ids, obs_ids))
# Jobs.subset_var(data, var_ids; kwargs...) =
# 	Job(create_spec(DataMatrixFunction(subset_matrix), data; kwargs..., var_ids))
# Jobs.subset_obs(data, obs_ids; kwargs...) =
# 	Job(create_spec(DataMatrixFunction(subset_matrix), data; kwargs..., obs_ids))


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




# # TODO: Use `:` instead of `nothing` and get rid of it at a preprocessing step after projecting
# function _filter_ind(f::Union{Var,Obs}, data; kwargs...)
# 	fun = get(kwargs, f isa Var ? :fvar : :fobs, nothing)
# 	fun === nothing && return nothing
# 	s = get_spec(f, data)
# 	project_ids = kwargs[f isa Var ? :project_var_ids : :project_obs_ids]
# 	id_spec = create_find_matching_ids_spec(fun, s; project_ids)
# 	create_ids_to_indices_spec(s, id_spec)
# end

# function filter_matrix(::Mat, data; kwargs...)
# 	matrix_spec = get_matrix_spec(data)
# 	var_ind = _filter_ind(Var(), data; kwargs...)
# 	obs_ind = _filter_ind(Obs(), data; kwargs...)
# 	if var_ind !== nothing && obs_ind !== nothing
# 		create_matrix_getindex_spec(matrix_spec; var_ind=prefetched(var_ind), obs_ind=prefetched(obs_ind))
# 	elseif var_ind !== nothing
# 		create_matrix_getindex_spec(matrix_spec; var_ind=prefetched(var_ind))
# 	elseif obs_ind !== nothing
# 		create_matrix_getindex_spec(matrix_spec; obs_ind=prefetched(obs_ind))
# 	else
# 		matrix_spec
# 	end
# end
# function filter_matrix(f::Union{Var,Obs}, data; kwargs...)
# 	s = get_spec(f, data)
# 	ind = _filter_ind(f, data; kwargs...)
# 	ind === nothing && return s
# 	create_annotation_getindex_spec(s, prefetched(ind))
# end


# # TODO: Use `:` instead of `nothing` and get rid of it at a preprocessing step after projecting
# function _filter_ind(f::Union{Var,Obs}, data; kwargs...)
# 	fun = get(kwargs, f isa Var ? :fvar : :fobs, :)
# 	fun === Colon() && return :

# 	s = get_spec(f, data)
# 	project_ids = kwargs[f isa Var ? :project_var_ids : :project_obs_ids]
# 	id_spec = create_find_matching_ids_spec(fun, s; project_ids)
# 	create_ids_to_indices_spec(s, id_spec)
# end

# function filter_matrix(::Mat, data; kwargs...)
# 	matrix_spec = get_matrix_spec(data)
# 	var_ind = _filter_ind(Var(), data; kwargs...)
# 	obs_ind = _filter_ind(Obs(), data; kwargs...)

# 	if var_ind !== nothing && obs_ind !== nothing
# 		create_matrix_getindex_spec(matrix_spec; var_ind=prefetched(var_ind), obs_ind=prefetched(obs_ind))
# 	elseif var_ind !== nothing
# 		create_matrix_getindex_spec(matrix_spec; var_ind=prefetched(var_ind))
# 	elseif obs_ind !== nothing
# 		create_matrix_getindex_spec(matrix_spec; obs_ind=prefetched(obs_ind))
# 	else
# 		matrix_spec
# 	end
# end
# function filter_matrix(f::Union{Var,Obs}, data; kwargs...)
# 	s = get_spec(f, data)
# 	ind = _filter_ind(f, data; kwargs...)
# 	ind === nothing && return s
# 	create_annotation_getindex_spec(s, prefetched(ind))
# end





# # Can we share more code with subset_ functions above?
# # Yes. By preprocessing.
# Jobs.filter_matrix(fvar, fobs, data; project_var_ids=:intersect, project_obs_ids=:no, kwargs...) =
# 	Job(create_spec(DataMatrixFunction(filter_matrix), data;  kwargs..., fvar, fobs, project_var_ids, project_obs_ids))
# Jobs.filter_var(fvar, data; project_ids=:intersect, kwargs...) =
# 	Job(create_spec(DataMatrixFunction(filter_matrix), data;  kwargs..., fvar, project_var_ids=project_ids))
# Jobs.filter_obs(fobs, data; project_ids=:no, kwargs...) =
# 	Job(create_spec(DataMatrixFunction(filter_matrix), data;  kwargs..., fobs, project_obs_ids=project_ids))


function _filter_ind(f, s; project_ids)
	# f === Colon() && return Colon() # THIS IS PROBABLY NOT CORRECT SINCE IT DOESN'T RESPECT project_ids
	id_spec = create_find_matching_ids_spec(f, s; project_ids)
	create_ids_to_indices_spec(s, id_spec)
end
# function filter_matrix(data; fvar=:, fobs=:, project_var_ids=:intersect, project_obs_ids=:no)
# 	var_ind = _filter_ind(fvar, get_var_spec(data); project_ids=project_var_ids)
# 	obs_ind = _filter_ind(fobs, get_obs_spec(data); project_ids=project_obs_ids)
# 	create_datamatrix_getindex_spec(data; var_ind, obs_ind)
# end
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


# Can we share more code with subset_ functions above?
# Yes. By preprocessing.
Jobs.filter_matrix(fvar, fobs, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fvar, fobs))
Jobs.filter_var(fvar, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fvar))
Jobs.filter_obs(fobs, data; kwargs...) =
	Job(create_spec(Preprocess(filter_matrix), data; kwargs..., fobs))
