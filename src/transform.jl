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
		create_spec(Projectable(logtransform_matrix), T, matrix_spec; var_ind, kwargs...)
	end
end
logtransform(::Obs, ::DataType, data; kwargs...) = get_obs_spec(data)


function Jobs.logtransform(T::DataType, counts; scale_factor=10_000, kwargs...)
	Job(create_spec(DataMatrixFunc(logtransform), T, counts; scale_factor, kwargs...))
end
Jobs.logtransform(counts; kwargs...) = Jobs.logtransform(Float64, counts; kwargs...)



# ------------------------------------------------------------------------------


function logcellcounts_impl(X, var_ind)
	feature_mask = falses(size(X,1))
	feature_mask[var_ind] .= true
	SCTransform.logcellcounts(X, feature_mask)
end


function logcellcounts(action::Action, X, var, var_ids; project_ids=:intersect)
	@assert project_ids in (:no,:yes,:intersect)

	X = action(X)

	if project_ids == :no
		var_ids2 = action(var_ids)
	elseif project_ids == :intersect
		var_ids2 = action(create_intersect_ids_spec(var_ids, var_ids))
	else#if project_ids == :yes
		vard_ids2 = var_ids
	end

	var_ind = create_ids_to_indices_spec(action(var), var_ids2) # TODO: Avoid using Projectable here
	create_spec(logcellcounts_impl, X, prefetch(var_ind); use_cache=true, __version=v"0.1.0")
end
create_logcellcounts_spec(X, var, var_ids; kwargs...) =
	create_spec(Projectable(logcellcounts), X, var, var_ids; kwargs...)


function scparams_impl(matrix; var_ind, log_cell_counts)
	feature_mask = falses(size(matrix,1))
	feature_mask[var_ind] .= true
	DataFrame(SCTransform.compute_scparams(matrix; log_cell_counts, feature_mask))
end
create_scparams_impl_spec(matrix; var_ind, log_cell_counts) =
	create_spec(scparams_impl, matrix; var_ind=prefetch(var_ind), log_cell_counts, use_cache=true, __version=v"0.1.0")


function scparams(action::Action, matrix, var, var_ids; log_cell_counts)
	# The inference is always done for the "eval" case
	var_ind = create_ids_to_indices_spec(var, var_ids) # TODO: Avoid using Projectable here
	params = create_scparams_impl_spec(matrix; var_ind, log_cell_counts) # DataFrame, but without IDs
	# params = create_annot_leftjoin_spec(var_ids, params) # TODO: Avoid using Projectable here
	params = create_hcat_spec(var_ids, params; copycols=false)

	if action isa Eval
		return params
	else#if actions is Projection
		# subset IDs
		var_ids_proj = create_spec(intersect_ids_impl, var_ids, action(var_ids); use_cache=true, __version=v"0.1.0")
		var_ind_proj = create_ids_to_indices_spec(params, var_ids_proj) # TODO: Avoid using Projectable here
		return create_annotation_getindex_spec(params, prefetch(var_ind_proj)) # TODO: Avoid using Projectable here
	end
end
create_scparams_spec(matrix, var, var_ids; log_cell_counts) =
	create_spec(Projectable(scparams), matrix, var, var_ids; log_cell_counts)



# counts[var_ind,:] must match params exactly
function sctransform_matrix(action::Action, T::DataType, counts, params;
                            var_ind,
                            clip=nothing, rtol=1e-3, atol=0.0,
                           )
	kwclip = clip===nothing ? (;) : (;clip)
	create_spec(SCPCore.sctransform_matrix, T, action(counts), action(params), action(var_ind); kwclip..., rtol, atol, use_cache=false, __version=v"0.1.0")
end
create_sctransform_matrix_spec(T, matrix, params; kwargs...) =
	create_spec(Projectable(sctransform_matrix), T, matrix, params; kwargs...)


function sctransform(f::Union{Mat,Var}, ::Type{T}, counts; var_filter=Returns(true), min_cells=5, annotate=false, kwargs...) where T
	matrix_spec = get_matrix_spec(counts)
	var_spec = get_var_spec(counts)

	var_ids_logcellcounts = create_find_matching_ids_spec(var_filter, var_spec; project_ids=:intersect)
	# var_ind_logcellcounts = create_ids_to_indices_spec(var_spec, var_ids_logcellcounts)
	# log_cell_counts = create_logcellcounts_spec(counts; var_ind=var_ind_logcellcounts) # maybe let the user pass project_ids onto here
	log_cell_counts = create_logcellcounts_spec(matrix_spec, var_spec, var_ids_logcellcounts) # maybe let the user pass project_ids onto here


	# min_cells
	nnz_cells = create_obs_counts_sum_impl_spec(!iszero, matrix_spec, :) # returns vector
	# var_nnz_cells = create_add_column_spec(get_var_ids(var_spec), "nnzCells", nnz_cells)
	var_nnz_cells = create_add_column_spec(create_get_ids_spec(var_spec), "nnzCells", nnz_cells)
	var_ids_min_cells = create_find_matching_ids_spec("nnzCells"=>>=(min_cells), var_nnz_cells; project_ids=:yes)
	var_ids = create_intersect_ids_spec(var_ids_logcellcounts, var_ids_min_cells)
	# var_ind = create_ids_to_indices_spec(var_spec, var_ids)

	# params_spec = create_scparams_spec(matrix_spec; var_ind, log_cell_counts)
	params_spec = create_scparams_spec(matrix_spec, var_spec, var_ids; log_cell_counts)


	# var_ids = create_find_matching_ids_spec(var_filter, var_spec; project_ids=:intersect)
	# var_ind = prefetch(create_ids_to_indices_spec(var_spec, var_ids))
	# params_spec = create_scparams_spec(matrix_spec, var_spec; var_ind) # TODO: pass min_cells if specified

	# params_ids = create_get_ids_spec(params_spec) # Get IDs from params_spec
	# intersected_var_ids = create_intersect_ids_spec(params_ids, var_ids) # Intersect IDs (using order from params_spec)
	# var_ind_out = prefetch(create_ids_to_indices_spec(var_spec, intersected_var_ids)) # Use the IDs to subset var annotations

	var_ind = create_ids_to_indices_spec(var_spec, var_ids)
	if f isa Var
		var_out = create_annotation_getindex_spec(var_spec, prefetch(var_ind))
		if annotate
			var_out = create_annot_leftjoin_spec(var_out, params_spec)
		end
		return var_out
	else # if f isa Mat
		return create_sctransform_matrix_spec(T, matrix_spec, params_spec; var_ind, kwargs...)
	end
end
sctransform(::Obs, ::DataType, counts; kwargs...) = get_obs_spec(counts)



function Jobs.sctransform(T::DataType, counts; kwargs...)
	Job(create_spec(DataMatrixFunc(sctransform), T, counts; kwargs...))
end
Jobs.sctransform(counts; kwargs...) = Jobs.sctransform(Float64, counts; kwargs...)






# function scparams_impl(matrix, var; var_ind, min_cells=5)
# 	@assert size(matrix,1) == size(var,1)
# 	feature_mask = falses(size(matrix,1))
# 	feature_mask[var_ind] .= true
# 	SCTransform.scparams(matrix, var; min_cells, feature_mask)
# end
# # NB: Currently uses the caching in SCTransform.jl, so we don't cache here.
# # create_scparams_spec(matrix, var; kwargs...) =
# # 	create_spec(scparams, matrix, var; kwargs..., use_cache=false, __version=v"0.1.0")

# function scparams(action::Action, matrix, var; kwargs...)
# 	params = create_spec(scparams_impl, matrix, var; kwargs..., use_cache=false, __version=v"0.1.0")

# 	if action isa Eval
# 		return params
# 	else#if action isa Projection
# 		# Maybe rewrite this without using Projectable specs. It would probably make it more clear.
# 		params_ids = create_spec(get_ids_impl, params; use_cache=false, __version=v"0.1.0")
# 		var_ids = create_get_ids_spec(var)
# 		intersected_var_ids = create_intersect_ids_spec(params_ids, var_ids) # Intersect IDs (using order from params)
# 		params_ind = create_ids_to_indices_spec(params, intersected_var_ids) # Use the IDs to subset params
# 		return create_spec(SCPCore.annotation_getindex, params, prefetch(action(params_ind)); use_cache=false, __version=v"0.1.0")
# 	end
# end
# create_scparams_spec(matrix, var; kwargs...) =
# 	create_spec(Projectable(scparams), matrix, var; kwargs...)



# # counts[var_ind,:] must match params exactly
# function sctransform_matrix(action::Action, T::DataType, counts, params, var_ind;
#                             clip=nothing, rtol=1e-3, atol=0.0,
# 	                       )
# 	kwclip = clip===nothing ? (;) : (;clip)
# 	create_spec(SCPCore.sctransform_matrix2, T, action(counts), params, var_ind; kwclip..., rtol, atol, use_cache=false, __version=v"0.1.0")
# end
# create_sctransform_matrix_spec(T, matrix, params; kwargs...) =
# 	create_spec(Projectable(sctransform_matrix), T, matrix_spec, params_spec; kwargs...)


# function sctransform(f::Union{Mat,Var}, T::DataType, counts; var_filter=Returns(true), kwargs...)
# 	matrix_spec = get_matrix_spec(counts)
# 	var_spec = get_var_spec(counts)

# 	var_ids = create_find_matching_ids_spec(var_filter, var_spec; project_ids=:intersect)
# 	var_ind = prefetch(create_ids_to_indices_spec(var_spec, var_ids))
# 	params_spec = create_scparams_spec(matrix_spec, var_spec; var_ind) # TODO: pass min_cells if specified

# 	params_ids = create_get_ids_spec(params_spec) # Get IDs from params_spec
# 	intersected_var_ids = create_intersect_ids_spec(params_ids, var_ids) # Intersect IDs (using order from params_spec)
# 	var_ind_out = prefetch(create_ids_to_indices_spec(var_spec, intersected_var_ids)) # Use the IDs to subset var annotations

# 	if f isa Var
# 		return create_annotation_getindex_spec(var_spec, prefetch(var_ind_out))
# 		# TODO: Add annotations from params_spec if requested
# 	else # if f isa Mat
# 		return create_sctransform_matrix_spec(T, matrix_spec, params_spec, var_ind_out; kwargs...)
# 	end
# end
# sctransform(::Obs, ::DataType, counts; kwargs...) = get_obs_spec(counts)



# function Jobs.sctransform(T::DataType, counts; kwargs...)
# 	Job(create_spec(DataMatrixFunc(sctransform), T, counts; kwargs...))
# end
# Jobs.sctransform(counts; kwargs...) = Jobs.sctransform(Float64, counts; kwargs...)





