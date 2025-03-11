# --- Internal checkpoints (asserts at the spec level) -------------------------
# function is_id_subset(a,b)
# 	id_col = only(names(a,1))
# 	@assert only(names(b,1)) == id_col
# 	@assert size(a,2) == 1
# 	@assert size(b,2) == 1

# 	issubset(a[!,1],b[!,1])
# end
# is_id_subset_spec(a, b) = create_spec(is_id_subset, a, b; use_cache=false, __version=v"0.1.0")

# function check_missing_ids_spec(value, ids, ids2)
# 	cond_spec = is_id_subset_spec(ids2, ids)
# 	error_spec = "Found missing IDs" # TODO: use spec so we can show names of missing IDs
# 	ifelse_spec(cond_spec, value, error_spec)
# 	# value
# end



function intersect_ids_impl(ids::DataFrame, ids2::DataFrame)
	id_col = only(names(ids,1))
	@assert only(names(ids2,1)) == id_col
	@assert size(ids,2) == 1
	@assert size(ids2,2) == 1
	DataFrame(id_col => intersect(ids[!,1],ids2[!,1]))
end

# function intersect_ids(action::Action, ids, ids2)
# 	ids = action(ids)
# 	ids2 = action(ids2)
# 	ids === ids2 && return ids

# 	spec = create_spec(intersect_ids_impl, ids, ids2; use_cache=false, __version=v"0.1.0")

# 	# Here's the place for checking for missing and extra IDs!
# 	check_missing_ids_spec(spec, ids, ids2)
# end
# create_intersect_ids_spec(ids, ids2) =
# 	create_spec(Projectable(intersect_ids), ids, ids2)



function find_matching_ids(action::Action, f, df; project_ids::Symbol)
	@assert project_ids in (:no,:yes,:intersect)
	if project_ids == :no
		f = action(f)
		df = action(df)
	end
	spec = create_spec(SCPCore.find_matching_ids, f, df; use_cache=true, __version=v"0.1.0")

	if project_ids == :intersect
		df = action(df)
		# TODO: simplify ids2 spec by using a function for extracting IDs directly
		ids2 = create_spec(SCPCore.find_matching_ids, Returns(true), df; use_cache=false, __version=v"0.1.0")
		spec = create_spec(intersect_ids_impl, spec, ids2; use_cache=true, __version=v"0.1.0")
	end
	spec
end


create_find_matching_ids_spec(f, df; project_ids) =
	create_spec(Projectable(find_matching_ids), f, df; project_ids)
Jobs.find_matching_ids(args...; kwargs...) =
	Job(create_find_matching_ids_spec(args...; kwargs...))




ids_to_indices(action::Action, args...) =
	create_spec(SCPCore.ids_to_indices, action(args)...; use_cache=true, __version=v"0.1.0")
create_ids_to_indices_spec(df, ids) =
	create_spec(Projectable(ids_to_indices), df, ids)

annotation_getindex(action::Action, args...) =
	create_spec(SCPCore.annotation_getindex, action(args)...; use_cache=true, __version=v"0.1.0")
create_annotation_getindex_spec(df, ind) =
	create_spec(Projectable(annotation_getindex), df, ind)



matrix_getindex(action::Action, args...; kwargs...) =
	create_spec(SCPCore.matrix_getindex, action(args)...; action(kwargs)..., use_cache=false, __version=v"0.1.0")
function create_matrix_getindex_spec(data; kwargs...)
	isempty(setdiff(keys(kwargs), (:var_ind,:obs_ind))) || throw(ArgumentError("Only allowed kwargs are `var_ind` and `obs_ind`, got: $(keys(kwargs))."))
	create_spec(Projectable(matrix_getindex), data; kwargs...)
end


extract_annotation(action::Action, args...) =
	create_spec(SCPCore.extract_annotation, action(args)...; use_cache=false, __version=v"0.1.0")
create_extract_annotation_spec(df, name) =
	create_spec(Projectable(extract_annotation), df, name)


annotation_nrows_impl(df) = size(df,1)
annotation_nrows(action::Action, df) =
	create_spec(annotation_nrows_impl, action(df); use_cache=false, __version=v"0.1.0")
annotation_nrows_spec(df) =
	create_spec(Projectable(annotation_nrows), df)


# TODO: Rename
hcat_impl(action::Action, args...) =
	create_spec(LinearAlgebra.hcat, action(args)...; use_cache=false, __version=v"0.1.0")
create_hcat_spec(args...) = create_spec(Projectable(hcat_impl), args...)



# NB: This assumes that the caller knows that `vals` exactly matches the ID column in `df`.
add_column_impl(df::DataFrame, name, vals) = insertcols(df, name=>vals; copycols=false)

add_column(action::Action, df, name, vals) =
	create_spec(add_column_impl, action(df), name, action(vals); use_cache=false, __version=v"0.1.0")
create_add_column_spec(df, name, vals) = create_spec(Projectable(add_column), df, name, vals)



prefixed_ids_impl(col::String, prefix::String, n::Int) = DataFrame(col=>string.(prefix, 1:n))
prefixed_ids(action::Action, col, prefix, n) =
	create_spec(prefixed_ids_impl, col, action(prefix), action(n); use_cache=false, __version=v"0.1.0")
create_prefixed_ids_spec(col, prefix, n) =
	create_spec(Projectable(prefixed_ids), col, prefix, n)
