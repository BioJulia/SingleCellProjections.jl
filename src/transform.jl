function logtransform_matrix(action::Action, T::DataType, matrix; scale_factor, var_ind)
	matrix = action(matrix)
	var_ind = action(var_ind)
	create_spec(SCPCore.logtransform_matrix, T, matrix; scale_factor, var_ind, use_cache=false, __version=v"0.1.0")
end

function logtransform(f::Union{Mat,Var}, T::DataType, data; var_filter=Returns(true), project_var_ids=:intersect, kwargs...)
	var_spec = get_var_spec(data)

	var_ids = create_find_matching_ids_spec(var_filter, var_spec; project_ids=project_var_ids)
	var_ind = prefetch(create_ids_to_indices_spec(var_spec, var_ids))

	if f isa Var
		create_annotation_getindex_spec(var_spec, var_ind)
	else # if f isa Mat
		matrix_spec = get_matrix_spec(data)
		create_spec(Projectable(logtransform_matrix), T, matrix_spec; use_cache=false, var_ind, kwargs...)
	end
end
logtransform(::Obs, ::DataType, data; kwargs...) = get_obs_spec(data)


function Jobs.logtransform(T::DataType, counts; scale_factor=10_000, kwargs...)
	Job(create_spec(DataMatrixFunc(logtransform), T, counts; use_cache=false, scale_factor, kwargs...))
end
Jobs.logtransform(counts; kwargs...) = Jobs.logtransform(Float64, counts; kwargs...)
