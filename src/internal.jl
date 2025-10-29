# WIP - some of these might not be needed.

ifelse_pr(action::Action, cond, x, y) = ifelse_spec(action(cond), action(x), action(y))
ifelse_pr_spec(cond, x, y) = create_spec(Projectable(ifelse_pr), cond, x, y)

args2vec_impl(::Type{T}, args...) where T = T[args...]
args2vec_pr(action::Action, ::Type{T}, args...) where T =
	create_spec(args2vec_impl, T, action(args)...; __version=v"0.1.0")
args2vec_spec(::Type{T}, args...) where T =
	create_spec(Projectable(args2vec_pr), T, args...)

getindex_impl_spec(v, ind) = create_spec(getindex, v, ind; __version=v"0.1.0")
getindex_pr(action, v, ind) = getindex_impl_spec(action(v), action(ind))
getindex_spec(v, ind) = create_spec(Projectable(getindex_pr), v, ind)

issubset_pr(action, a, b) = create_spec(issubset, action(a), action(b); __version=v"0.1.0")
issubset_spec(a, b) = create_spec(Projectable(issubset_pr), a, b)

setdiff_pr(action, a, b) = create_spec(setdiff, action(a), action(b); __version=v"0.1.0")
setdiff_spec(a, b) = create_spec(Projectable(setdiff_pr), a, b)

intersect_impl_spec(a, b, args...) = create_spec(intersect, a, b, args...; __version=v"0.1.0")
intersect_pr(action::Action, a, b, args...) = intersect_impl_spec(action(a), action(b), action(args)...)
intersect_spec(a, b, args...) = create_spec(Projectable(intersect_pr), a, b, args...)

length_pr(action, x) = create_spec(length, action(x); __version=v"0.1.0")
length_spec(x) = create_spec(Projectable(length_pr), x)

isequal_pr(action, x, y) = create_spec(isequal, action(x), action(y); __version=v"0.1.0")
isequal_spec(x, y) = create_spec(Projectable(isequal_pr), x, y)








nvar_spec(data) = table_nrow_spec(get_var_spec(data))
Jobs.nvar(data) = Job(nvar_spec(data))

nobs_spec(data) = table_nrow_spec(get_obs_spec(data))
Jobs.nobs(data) = Job(nobs_spec(data))




index_isnoop_spec(ind, n) =
	create_spec(SCPCore.index_isnoop, ind, n; __version=v"0.0.1")
simplify_ind_spec(ind, n) =
	ind === Colon() ? Colon() : ifelse_spec(index_isnoop_spec(ind, n), Colon(), ind) # early out if is already known to be Colon








function find_matching_ids(action::Action, f, df; project_ids::Symbol)
	# TODO: Consider having a simplified path when indexing with :
	# We just need to handle different project_ids cases properly
	# if f === : and project_ids != intersect
	# 	- just return :
	#	- or return the get_id_spec? feels wasteful to call find_matching_ids
	# otherwise do what we do now

	@assert project_ids in (:no, :yes, :intersect)
	if project_ids == :no
		f = action(f)
		df = action(df)
	end

	# TODO: If `f` is a pair, we can subset the columns of df to avoid involving them in the call
	ind = cached(create_spec(SCPCore.find_matching_ind, f, df; __version=v"0.1.2"))
	matching_ids = table_getindex_spec(id_column_spec(df), ind)

	if project_ids == :intersect && action isa Projection
		# df = action(df)
		# TODO: simplify ids2 spec by using a function for extracting IDs directly
		# ids2 = create_spec(SCPCore.find_matching_ids, Returns(true), df; __version=v"0.1.0")
		# spec = cached(create_spec(intersect_ids_impl, spec, ids2; __version=v"0.1.0"))

		# ids2 = id_column_spec(df)
		# spec = cached(create_spec(intersect_ids_impl, spec, action(ids2); __version=v"0.1.0"))

		# ids2 = action(column_data_spec(df,1))
		# matching_ids = cached(intersect_impl_spec(matching_ids, ids2))

		ids2 = action(id_column_spec(df))
		matching_ids = cached(intersect_ids_spec(matching_ids, ids2)) # Use order from unprojected
	end
	matching_ids
end


create_find_matching_ids_spec(f, df; project_ids) =
	create_spec(Projectable(find_matching_ids), f, df; project_ids)
Jobs.find_matching_ids(args...; kwargs...) =
	Job(create_find_matching_ids_spec(args...; kwargs...))




ids_to_indices(action::Action, df, ids) =
	cached(create_spec(SCPCore.ids_to_indices, action(df), action(ids); __version=v"0.1.2"))
create_ids_to_indices_spec(df, ids) =
	create_spec(Projectable(ids_to_indices), df, ids)




matrix_getindex_impl(matrix; kwargs...) =
	create_spec(SCPCore.matrix_getindex, matrix; kwargs..., __version=v"0.1.0")

function matrix_getindex_pre(matrix; var_ind, obs_ind)
	if var_ind == Colon() && obs_ind == Colon()
		matrix
	else
		matrix_getindex_impl(matrix; var_ind, obs_ind)
	end
end

function matrix_getindex_pr(action::Action, matrix; var_ind, obs_ind, nvar=nothing, nobs=nothing)
	matrix = action(matrix)
	var_ind = action(var_ind)
	obs_ind = action(obs_ind)
	nvar !== nothing && (var_ind = simplify_ind_spec(var_ind, nvar))
	nobs !== nothing && (obs_ind = simplify_ind_spec(obs_ind, nobs))
	var_ind = fetched(var_ind)
	obs_ind = fetched(obs_ind)
	create_spec(Preprocess(matrix_getindex_pre), matrix; var_ind, obs_ind)
end

function create_matrix_getindex_spec(matrix; var_ind=:, obs_ind=:, kwargs...)
	create_spec(Projectable(matrix_getindex_pr), matrix; var_ind, obs_ind, kwargs...)
end



datamatrix_getindex(::Mat, data; kwargs...) =
	create_matrix_getindex_spec(get_matrix_spec(data); nvar=nvar_spec(data), nobs=nobs_spec(data), kwargs...)
datamatrix_getindex(::Var, data; var_ind=:, kwargs...) =
	table_getindex_spec(get_var_spec(data), var_ind)
datamatrix_getindex(::Obs, data; obs_ind=:, kwargs...) =
	table_getindex_spec(get_obs_spec(data), obs_ind)


create_datamatrix_getindex_spec(data; kwargs...) =
	create_spec(DataMatrixFunction(datamatrix_getindex), data; kwargs...)



extract_annotation(action::Action, args...) =
	create_spec(SCPCore.extract_annotation, action(args)...; __version=v"0.1.0")
create_extract_annotation_spec(df, name) =
	create_spec(Projectable(extract_annotation), df, name)


# DEPRECATED: Use `get_value_colname_spec` instead
function annotation_name_impl(df)
	@assert ncol(df)==2
	only(names(df, 2))
end
annotation_name(action::Action, df) = create_spec(annotation_name_impl, action(df); __version=v"0.1.0")
annotation_name_spec(df) = create_spec(Projectable(annotation_name), df)



# TODO: Rename?
hcat_impl(action::Action, args...; kwargs...) =
	create_spec(hcat, action(args)...; kwargs..., __version=v"0.1.0")
create_hcat_spec(args...; kwargs...) = create_spec(Projectable(hcat_impl), args...)





prefixed_ids_impl(col::String, prefix::String, n::Int) = DataFrame(col=>string.(prefix, 1:n))
prefixed_ids(action::Action, col, prefix, n) =
	create_spec(prefixed_ids_impl, col, action(prefix), action(n); __version=v"0.1.0")
create_prefixed_ids_spec(col, prefix, n) =
	create_spec(Projectable(prefixed_ids), col, prefix, n)
