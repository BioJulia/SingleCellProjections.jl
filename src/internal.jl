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

length_pr(action, x) = create_spec(length, action(x); __version=v"0.1.0")
length_spec(x) = create_spec(Projectable(length_pr), x)

isequal_pr(action, x, y) = create_spec(isequal, action(x), action(y); __version=v"0.1.0")
isequal_spec(x, y) = create_spec(Projectable(isequal_pr), x, y)









annotation_nrow_impl(df) = size(df,1)
annotation_nrow(action::Action, df) =
	create_spec(annotation_nrow_impl, action(df); __version=v"0.1.0")
annotation_nrow_spec(df) =
	create_spec(Projectable(annotation_nrow), df)


datamatrix_nvar_spec(data) = annotation_nrow_spec(get_var_spec(data))
datamatrix_nobs_spec(data) = annotation_nrow_spec(get_obs_spec(data))





# TESTING
index_isnoop_spec(ind, n) =
	create_spec(SCPCore.index_isnoop, ind, n; __version=v"0.0.1")
# simplify_ind_spec(ind, n) =
# 	ReproducibleJobs.ifelse_spec(index_isnoop_spec(ind, n), Colon(), ind)
simplify_ind_spec(ind, n) =
	ind === Colon() ? Colon() : ReproducibleJobs.ifelse_spec(index_isnoop_spec(ind, n), Colon(), ind) # early out if is already known to be Colon





function intersect_ids_impl(ids::DataFrame, ids2::DataFrame)
	id_col = only(names(ids,1))
	@assert only(names(ids2,1)) == id_col
	@assert size(ids,2) == 1
	@assert size(ids2,2) == 1
	DataFrame(id_col => intersect(ids[!,1],ids2[!,1]))
end

function intersect_ids(action::Action, ids, ids2; fix_first=true, fix_second=false)
	ids = fix_first ? ids : action(ids)
	ids2 = fix_second ? ids2 : action(ids2)
	ids === ids2 && return ids
	create_spec(intersect_ids_impl, ids, ids2; __version=v"0.1.0")
end
create_intersect_ids_spec(ids, ids2; kwargs...) =
	create_spec(Projectable(intersect_ids), ids, ids2; kwargs...)




get_ids_impl(df) = select(df, 1; copycols=false)
get_ids(action::Action, df) = create_spec(get_ids_impl, action(df); __version=v"0.1.0")
create_get_ids_spec(df) = create_spec(Projectable(get_ids), df)



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
	spec = cached(create_spec(SCPCore.find_matching_ids, f, df; __version=v"0.1.0"))

	if project_ids == :intersect && action isa Projection
		# df = action(df)
		# TODO: simplify ids2 spec by using a function for extracting IDs directly
		# ids2 = create_spec(SCPCore.find_matching_ids, Returns(true), df; __version=v"0.1.0")
		# spec = cached(create_spec(intersect_ids_impl, spec, ids2; __version=v"0.1.0"))

		ids2 = create_get_ids_spec(df)
		spec = cached(create_spec(intersect_ids_impl, spec, action(ids2); __version=v"0.1.0"))
	end
	spec
end


create_find_matching_ids_spec(f, df; project_ids) =
	create_spec(Projectable(find_matching_ids), f, df; project_ids)
Jobs.find_matching_ids(args...; kwargs...) =
	Job(create_find_matching_ids_spec(args...; kwargs...))




ids_to_indices(action::Action, df, ids) =
	cached(create_spec(SCPCore.ids_to_indices, action(df), action(ids); __version=v"0.1.2"))
create_ids_to_indices_spec(df, ids) =
	create_spec(Projectable(ids_to_indices), df, ids)

# # TODO: Use `:` instead of `nothing` and get rid of it at a preprocessing step after projecting
# annotation_getindex(action::Action, df, ind) =
# 	create_spec(SCPCore.annotation_getindex, action(df), action(ind); __version=v"0.1.0")
# create_annotation_getindex_spec(df, ind) =
# 	create_spec(Projectable(annotation_getindex), df, ind)

annotation_getindex_impl(df, ind) =
	create_spec(SCPCore.annotation_getindex, df, ind; __version=v"0.1.0")
annotation_getindex_pre(df, ind) =
	ind === Colon() ? df : annotation_getindex_impl(df, ind)
function annotation_getindex_pr(action::Action, df, ind)
	df = action(df)
	ind = action(ind)
	ind = simplify_ind_spec(ind, annotation_nrow_spec(df))
	create_spec(Preprocess(annotation_getindex_pre), df, fetched(ind))
end
create_annotation_getindex_spec(df, ind) =
	create_spec(Projectable(annotation_getindex_pr), df, ind)



# matrix_getindex(action::Action, args...; kwargs...) =
# 	create_spec(SCPCore.matrix_getindex, action(args)...; action(kwargs)..., __version=v"0.1.0")
# function create_matrix_getindex_spec(data; kwargs...)
# 	isempty(setdiff(keys(kwargs), (:var_ind,:obs_ind))) || throw(ArgumentError("Only allowed kwargs are `var_ind` and `obs_ind`, got: $(keys(kwargs))."))
# 	create_spec(Projectable(matrix_getindex), data; kwargs...)
# end


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



# TODO: Use `:` instead of `nothing` and get rid of it at a preprocessing step after projecting
# datamatrix_getindex(::Mat, data; kwargs...) = create_matrix_getindex_spec(get_matrix_spec(data); kwargs...)
datamatrix_getindex(::Mat, data; kwargs...) =
	create_matrix_getindex_spec(get_matrix_spec(data); nvar=datamatrix_nvar_spec(data), nobs=datamatrix_nobs_spec(data), kwargs...)
# function datamatrix_getindex(::Var, data; var_ind=nothing, kwargs...)
# 	var_spec = get_var_spec(data)
# 	var_ind === nothing ? var_spec : create_annotation_getindex_spec(var_spec, var_ind)
# end
# function datamatrix_getindex(::Obs, data; obs_ind=nothing, kwargs...)
# 	obs_spec = get_obs_spec(data)
# 	obs_ind === nothing ? obs_spec : create_annotation_getindex_spec(obs_spec, obs_ind)
# end
datamatrix_getindex(::Var, data; var_ind=:, kwargs...) =
	create_annotation_getindex_spec(get_var_spec(data), var_ind)
datamatrix_getindex(::Obs, data; obs_ind=:, kwargs...) =
	create_annotation_getindex_spec(get_obs_spec(data), obs_ind)


create_datamatrix_getindex_spec(data; kwargs...) =
	create_spec(DataMatrixFunction(datamatrix_getindex), data; kwargs...)



extract_annotation(action::Action, args...) =
	create_spec(SCPCore.extract_annotation, action(args)...; __version=v"0.1.0")
create_extract_annotation_spec(df, name) =
	create_spec(Projectable(extract_annotation), df, name)



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

# vcat_impl(action::Action, args...; kwargs...) =
# 	create_spec(vcat, action(args)...; kwargs..., __version=v"0.1.0")
# create_vcat_spec(args...; kwargs...) = create_spec(Projectable(vcat_impl), args...)



# NB: This assumes that the caller knows that `vals` exactly matches the ID column in `df`.
add_column_impl(df::DataFrame, name, vals) = insertcols(df, name=>vals; copycols=false)

add_column(action::Action, df, name, vals) =
	create_spec(add_column_impl, action(df), name, action(vals); __version=v"0.1.0")
create_add_column_spec(df, name, vals) = create_spec(Projectable(add_column), df, name, vals)



prefixed_ids_impl(col::String, prefix::String, n::Int) = DataFrame(col=>string.(prefix, 1:n))
prefixed_ids(action::Action, col, prefix, n) =
	create_spec(prefixed_ids_impl, col, action(prefix), action(n); __version=v"0.1.0")
create_prefixed_ids_spec(col, prefix, n) =
	create_spec(Projectable(prefixed_ids), col, prefix, n)
