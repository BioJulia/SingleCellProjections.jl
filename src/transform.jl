function logtransform(f::Union{Mat,Var}, T::DataType, data; var_filter=:, project_var_ids=:intersect, scale_factor)
	var_spec = get_var_spec(data)
	var_ind = prefetched(create_find_matching_ind_spec(var_filter, var_spec; project_ids=project_var_ids))

	if f isa Var
		table_getindex_spec(var_spec, var_ind)
	else # if f isa Mat
		matrix_spec = get_matrix_spec(data)
		create_spec(SCPCore.logtransform_matrix, T, matrix_spec; var_ind, scale_factor, __version=v"0.1.0")
	end
end
logtransform(::Obs, ::DataType, data; kwargs...) = get_obs_spec(data)


function Jobs.logtransform(T::DataType, counts; scale_factor=10_000, kwargs...)
	Job(create_spec(DataMatrixFunction(logtransform), T, counts; scale_factor, kwargs...))
end
Jobs.logtransform(counts; kwargs...) = Jobs.logtransform(Float64, counts; kwargs...)



# ------------------------------------------------------------------------------


function logcellcounts_impl(X, var_ind)
	feature_mask = falses(size(X,1))
	feature_mask[var_ind] .= true
	SCTransform.logcellcounts(X, feature_mask)
end
logcellcounts_spec(X, var_ind) = create_spec(logcellcounts_impl, X, var_ind; __version=v"0.1.0")


function scparams_impl(matrix; var_ind, log_cell_counts)
	feature_mask = falses(size(matrix,1))
	feature_mask[var_ind] .= true
	df = DataFrame(SCTransform.compute_scparams(matrix; log_cell_counts, feature_mask); copycols=false)
	table_to_compound_result(df)
end
create_scparams_impl_spec(matrix; var_ind, log_cell_counts) =
	create_spec(scparams_impl, matrix; var_ind=prefetched(var_ind), log_cell_counts, __version=v"0.1.1")


function scparams(action::Action, matrix, var, var_ind; log_cell_counts)
	# The inference is always done for the "eval" case
	params = create_scparams_impl_spec(matrix; var_ind, log_cell_counts) # DataFrame, but without IDs
	params = table_from_compound_result(params)

	if action isa Eval
		return params
	else#if actions is Projection
		# We need to remap IDs
		var_ids = id_column_spec(var)
		var_ids2 = action(var_ids)

		param_ids = table_getindex_spec(var_ids, var_ind) # The IDs represented in the params table
		var_ids_proj = intersect_ids_spec(param_ids, var_ids2)
		var_ind_proj = indexin_spec(var_ids_proj, param_ids; not_found=:error)
		return table_getindex_spec(params, prefetched(var_ind_proj))
	end
end
create_scparams_spec(matrix, var, var_ind; log_cell_counts) =
	create_spec(Projectable(scparams), matrix, var, var_ind; log_cell_counts)



# counts[var_ind,:] must match params exactly
function sctransform_matrix_spec(T, matrix, params;
                                 var_ind,
                                 clip=nothing, rtol=1e-3, atol=0.0)
	kwclip = clip===nothing ? (;) : (;clip)
	create_spec(SCPCore.sctransform_matrix, T, matrix, params, var_ind; kwclip..., rtol, atol, __version=v"0.1.0")
end



function sctransform(f::Union{Mat,Var}, ::Type{T}, counts; var_filter=:, min_cells=5, annotate=false, kwargs...) where T
	matrix_spec = get_matrix_spec(counts)
	var_spec = get_var_spec(counts)

	var_ind_logcellcounts = prefetched(create_find_matching_ind_spec(var_filter, var_spec; project_ids=:intersect))
	log_cell_counts = logcellcounts_spec(matrix_spec, var_ind_logcellcounts)

	# min_cells
	nnz_cells = cached(counts_sum_impl_spec(!iszero, matrix_spec, :; dims=2)) # returns vector
	var_nnz_cells = add_column_spec(id_column_spec(var_spec), "nnzCells", nnz_cells)
	var_ind_min_cells = create_find_matching_ind_spec("nnzCells"=>>=(min_cells), var_nnz_cells; project_ids=:yes)

	var_ind = prefetched(intersect_ind_spec(var_ind_logcellcounts, var_ind_min_cells))
	params_spec = create_scparams_spec(matrix_spec, var_spec, var_ind; log_cell_counts)

	if f isa Var
		var_out = table_getindex_spec(var_spec, var_ind)
		if annotate
			var_out = table_hcat_spec(var_out, params_spec)
		end
		return var_out
	else # if f isa Mat
		return sctransform_matrix_spec(T, matrix_spec, params_spec; var_ind, kwargs...)
	end
end
sctransform(::Obs, ::DataType, counts; kwargs...) = get_obs_spec(counts)



function Jobs.sctransform(T::DataType, counts; kwargs...)
	Job(create_spec(DataMatrixFunction(sctransform), T, counts; kwargs...))
end
Jobs.sctransform(counts; kwargs...) = Jobs.sctransform(Float64, counts; kwargs...)
