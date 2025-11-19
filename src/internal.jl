# WIP - some of these might not be needed.

# This must be used instead of ifelse_spec when `cond` should be projected. Otherwise it will be fetched **before** projections take place.
ifelse_pr(action::Action, cond, x, y) = ifelse_spec(action(cond), action(x), action(y))
ifelse_pr_spec(cond, x, y) = create_spec(Projectable(ifelse_pr), cond, x, y)

_getindex_error(ind) = throw(ArgumentError("Raw indices not allowed when projecting (unless containers are identical). Got indices: $ind."))
_getindex_error_spec(ind) = create_spec(_getindex_error, ind; __version=v"0.1.0")

getindex_impl(::Preprocessing, v, ind) = ind===Colon() ? v : create_spec(getindex, v, ind; __version=v"0.1.0")
getindex_impl_spec(v, ind) = create_spec(Preprocess(getindex_impl), v, fetched(ind))

function getindex_pr(action, v, ind)
	v_p = action(v)
	result = getindex_impl_spec(v_p, action(ind))

	if action isa Projection && !(ind isa Spec)
		cond = isequal_impl_spec(v, v_p)
		result = ifelse_spec(cond, result, _getindex_error_spec(ind))
	end

	result
end
getindex_spec(v, ind) = create_spec(Projectable(getindex_pr), v, ind)



# TODO: Find a better name? These are essentially getindex, but we map `nothing` to `missing`. It is useful for leftjoins on IDs.
function getindex_or_missing(v::AbstractVector{Tv}, ind::AbstractVector{Ti}) where {Tv,Ti}
	if Nothing<:Ti
		# If we want, we could narrow the type to T if there are no Nothing in the vector
		# But we do that work in indexin (narrow the index type), so skip it here, at least for now.
		Union{Missing,Tv}[i===nothing ? missing : v[i] for i in ind]
	else
		v[ind]
	end
end

getindex_or_missing_impl(::Preprocessing, v, ind) = ind===Colon() ? v : create_spec(getindex_or_missing, v, ind; __version=v"0.1.1")
getindex_or_missing_impl_spec(v, ind) = create_spec(Preprocess(getindex_or_missing_impl), v, fetched(ind))

function getindex_or_missing_pr(action, v, ind)
	v_p = action(v)
	result = getindex_or_missing_impl_spec(v_p, action(ind))

	if action isa Projection && !(ind isa Spec)
		cond = isequal_impl_spec(v, v_p)
		result = ifelse_spec(cond, result, _getindex_or_missing_error_spec(ind))
	end

	result
end
getindex_or_missing_spec(v, ind) = create_spec(Projectable(getindex_or_missing_pr), v, ind)



intersect_spec(a, b, args...) = create_spec(intersect, a, b, args...; __version=v"0.1.0")
hcat_spec(args...; kwargs...) = create_spec(hcat, args...; kwargs..., __version=v"0.1.0")
length_spec(x) = create_spec(length, x; __version=v"0.1.0")



function intersect_ind_impl(a, b)
	a == Colon() && return b
	b == Colon() && return a
	intersect(a,b)
end
function intersect_ind(::Preprocessing{E}, a, b) where E
	a == Colon() && return b
	b == Colon() && return a
	E && return create_spec(Preprocess{false}(intersect_ind), a, b)
	return create_spec(intersect_ind_impl, a, b; __version=v"0.1.0")
end

"""
	intersect_ind_spec(a, b)

Create a spec to compute the intersection of `Vector`s `a` and `b` with indexes.
Just like `intersect`, but allows `a` and/or `b` to be `:`.
"""
intersect_ind_spec(a, b) = create_spec(Preprocess(intersect_ind), a, b)







function isequal_pre(::Preprocessing{E}, x, y) where E
	if isequal(x, y)
		true # early out
	elseif !(x isa Spec) && !(y isa Spec)
		false # early out
	elseif E
		create_spec(Preprocess{false}(isequal_pre), x, y)
	else
		create_spec(isequal, x, y; __version=v"0.1.0")
	end
end
isequal_spec(x, y) = create_spec(Preprocess(isequal_pre), x, y)


function indexin_impl(a::AbstractVector, b::AbstractVector; not_found)
	# TODO: Find better names `not_found`, `:error`, `:skip` and `:nothing`
	@assert not_found in (:error, :skip, :nothing)
	ind = indexin(a, b)

	if not_found == :error
		any(isnothing, ind) && error("Found $(count(isnothing,ind)) values in `a` that are not present in `b`.")
	elseif not_found == :skip
		all(isnothing, ind) && @warn "indexin_impl: No values in `a` were present in `b`."
		filter!(!isnothing, ind)
	elseif not_found == :nothing && any(isnothing, ind)
		return ind
	end

	ind = convert(Vector{Int}, ind)
	if ind == 1:length(b)
		return Colon() # simplify common case
	else
		return ind
	end
end

function indexin_impl(a::DataFrame, b::DataFrame; not_found)
	@assert ncol(a)==1
	@assert ncol(b)==1
	@assert only(names(a,1)) == only(names(b,1))
	indexin_impl(a[!,1], b[!,1]; not_found)
end

indexin_spec(a, b; not_found=:error) = create_spec(indexin_impl, a, b; not_found, __version=v"0.1.2")






nvar_spec(data) = table_nrow_spec(get_var_spec(data))
Jobs.nvar(data) = Job(nvar_spec(data))

nobs_spec(data) = table_nrow_spec(get_obs_spec(data))
Jobs.nobs(data) = Job(nobs_spec(data))


find_matching_ind_impl_spec(f, df) = create_spec(SCPCore.find_matching_ind, f, df; __version=v"0.1.4")


function find_matching_ind(action::Action, f, df; project_ids::Symbol)
	@assert project_ids in (:no, :yes, :intersect)
	if project_ids == :no
		f = action(f)
		df = action(df)
	end

	# TODO: If `f` is a pair, we can subset the columns of df to avoid involving them in the call
	if f === Colon()
		matching_ind = Colon()
	elseif f isa AbstractRange
		matching_ind = f
	elseif f isa Pair
		k = first(f)

		# subset the columns to only depend on those that are used
		if k isa AbstractString
			x = get_columns_spec(df, k)
			matching_ind = cached(find_matching_ind_impl_spec(f, x))
		elseif k isa AbstractVector
			x = get_columns_spec(df, k...)
			matching_ind = cached(find_matching_ind_impl_spec(f, x))
		elseif k isa Union{Spec, DataFrame}
			# k is an "Annotation" - a DataFrame with an ID and a value column. Will be leftjoined and the function will be applied to the leftjoined vector with values.

			ids_a = id_column_spec(df)
			ids_b = id_column_spec(k)
			ind_spec = indexin_spec(ids_a, ids_b; not_found=:nothing)
			v = value_column_data_spec(k)
			x = getindex_or_missing_spec(v, ind_spec) # The values of the annotation `k`, reordered to match the order in df.

			matching_ind = cached(find_matching_ind_impl_spec(last(f), x))
		else
			throw(ArgumentError("Unknown column selector $k of type $(typeof(k))."))
		end
	else
		# This is used when `f` is a function taking a DataFrameRow
		matching_ind = cached(find_matching_ind_impl_spec(f, df))
	end

	if action isa Eval || project_ids == :no
		return matching_ind
	else
		# We need to remap the indices, going through IDs
		ids = id_column_spec(df)
		ids2 = action(ids) # IDs from projected dataset

		cond = isequal_spec(ids, ids2)

		# matching_ids = table_getindex_impl_spec(ids, matching_ind) # unprojected IDs (NB: this will simplify if matching_ind==Colon())
		matching_ids = table_getindex_spec(ids, matching_ind) # unprojected IDs (NB: this will simplify if matching_ind==Colon())
		if project_ids == :yes
			proj_ind = indexin_spec(matching_ids, ids2; not_found=:error) # Use order from unprojected
		else#if project_ids == :intersect
			proj_ind = indexin_spec(matching_ids, ids2; not_found=:skip) # Use order from unprojected
		end

		# This gives us an early out when ids==ids2, since we can just return matching_ind in that case (no need to bother with getting matching ids and doing indexin)
		ifelse_spec(cond, matching_ind, proj_ind)
	end
end
create_find_matching_ind_spec(f, df; project_ids) =
	create_spec(Projectable(find_matching_ind), f, df; project_ids)
# Jobs.find_matching_ind(args...; kwargs...) =
# 	Job(create_find_matching_ind_spec(args...; kwargs...))





# Do we need these, or upstream steps always return Colon() when they can?
index_isnoop_spec(ind, n) =
	create_spec(SCPCore.index_isnoop, ind, n; __version=v"0.0.1")
simplify_ind_spec(ind, n) =
	ind === Colon() ? Colon() : ifelse_spec(index_isnoop_spec(ind, n), Colon(), ind) # early out if is already known to be Colon








matrix_getindex_impl(matrix; kwargs...) =
	create_spec(SCPCore.matrix_getindex, matrix; kwargs..., __version=v"0.1.0")

function matrix_getindex_pre(::Preprocessing, matrix; var_ind, obs_ind)
	if var_ind == Colon() && obs_ind == Colon()
		matrix
	else
		matrix_getindex_impl(matrix; var_ind, obs_ind)
	end
end



function _matrix_ind_spec(action::Action, ind, n=nothing)
	ind_p = action(ind)
	if action isa Projection && !(ind isa Spec)
		cond = isequal_impl_spec(ind, ind_p)
		ind_p = ifelse_spec(cond, ind_p, _getindex_error_spec(ind))
	end
	n !== nothing && (ind_p = simplify_ind_spec(ind_p, n))
	ind_p = fetched(ind_p)
end

function matrix_getindex_pr(action::Action, matrix; var_ind, obs_ind, nvar=nothing, nobs=nothing)
	matrix = action(matrix)
	var_ind = _matrix_ind_spec(action, var_ind, nvar)
	obs_ind = _matrix_ind_spec(action, obs_ind, nobs)
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





prefixed_id_values(prefix::String, n) = string.(prefix, 1:n)
function prefixed_ids(::Preprocessing, col::String, prefix, n)
	col_data = create_spec(prefixed_id_values, prefix, n; __version=v"0.1.0")
	create_table_spec(col=>col_data)
end
prefixed_ids_spec(col, prefix, n) =
	create_spec(Preprocess(prefixed_ids), col, prefix, n)
