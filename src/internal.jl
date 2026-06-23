# WIP - some of these might not be needed.

# This must be used instead of ifelse_job when `cond` should be projected. Otherwise it will be fetched **before** projections take place.
ifelse_pr(action::Action, cond, x, y) = ifelse_job(action(cond), action(x), action(y))
ifelse_pr_job(cond, x, y) = create_job(Projectable(ifelse_pr), cond, x, y)


# Find a better name?
function combine_vectors_impl(args...; delim=nothing)
	if delim !== nothing
		n = length(args)
		args = Iterators.take(Iterators.flatten(zip(args, Iterators.cycle(delim))), 2n-1) # insert delim between
	end
	string.(args...)
end
combine_vectors_job(args...; kwargs...) = create_job(combine_vectors_impl, args...; kwargs..., __version=v"0.1.0")





_getindex_error(ind) = throw(ArgumentError("Raw indices not allowed when projecting (unless containers are identical). Got indices: $ind."))
_getindex_error_job(ind) = create_job(_getindex_error, ind; __version=v"0.1.0")

# getindex_impl(::Preprocessing, v, ind) = ind===Colon() ? v : create_job(getindex, v, ind; __version=v"0.1.0")
# function getindex_impl(::Preprocessing{E}, v, ind) where E
# 	if ind === Colon()
# 		v
# 	elseif E
# 		create_job(Preprocess{false}(getindex_impl), v, fetched(ind)) # NB: This way we fetch after projections are handled!
# 	else
# 		create_job(getindex, v, ind; __version=v"0.1.0")
# 	end
# end

function getindex_impl(::Preprocessing, v, ind)
	if ind === Colon()
		v # Projections have been handled, so indexing by `:` is OK
	elseif v isa SpecRef && v.f === getindex
		# Collapse nested getindex calls which is important for getting canonical representations
		create_job(getindex, v.args[1], compose_ind(v.args[2], ind); __version=v"0.1.0")
	else
		create_job(getindex, v, ind; __version=v"0.1.0")
	end
end
getindex_impl_job(v, ind) = create_job(Preprocess{false}(getindex_impl), v, fetched(ind))

function getindex_pr(action, v, ind)
	v_p = action(v)
	result = getindex_impl_job(v_p, action(ind))

	if action isa Projection && !(ind isa SpecRef) # TODO: Fix, this will trigger even if ind is replaced by the action, which it shouldn't - maybe hard to avoid?
		cond = isequal_job(v, v_p)
		result = ifelse_job(cond, result, _getindex_error_job(ind))
	end

	result
end
getindex_job(v, ind) = create_job(Projectable(getindex_pr), v, ind)



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

getindex_or_missing_impl(::Preprocessing, v, ind) = ind===Colon() ? v : create_job(getindex_or_missing, v, ind; __version=v"0.1.1")
getindex_or_missing_impl_job(v, ind) = create_job(Preprocess(getindex_or_missing_impl), v, fetched(ind))

function getindex_or_missing_pr(action, v, ind)
	v_p = action(v)
	result = getindex_or_missing_impl_job(v_p, action(ind))

	if action isa Projection && !(ind isa SpecRef)
		cond = isequal_job(v, v_p)
		result = ifelse_job(cond, result, _getindex_or_missing_error_job(ind)) # TODO: Fix _getindex_or_missing_error_job isn't defined anywhere...
	end

	result
end
getindex_or_missing_job(v, ind) = create_job(Projectable(getindex_or_missing_pr), v, ind)



intersect_job(a, b, args...) = create_job(intersect, a, b, args...; __version=v"0.1.0")
length_job(x) = create_job(length, x; __version=v"0.1.0")
unique_job(x) = create_job(unique, x; __version=v"0.1.0")
join_job(x, args...) = create_job(join, x, args...; __version=v"0.1.0")
reshape_job(A, args...) = create_job(reshape, A, args...; __version=v"0.1.0")
repeat_job(A, args...; kwargs...) = create_job(repeat, A, args...; kwargs..., __version=v"0.1.0")
prod_job(args...; kwargs...) = create_job(prod, args...; kwargs..., __version=v"0.1.0")
allequal_job(x) = create_job(allequal, x; __version=v"0.1.0")

# vcat_job(args...; kwargs...) = create_job(vcat, args...; kwargs..., __version=v"0.1.0")
# hcat_job(args...; kwargs...) = create_job(hcat, args...; kwargs..., __version=v"0.1.0")

vcat_impl(v; kwargs...) = reduce(vcat, v; kwargs...)
vcat_job(v; kwargs...) = create_job(vcat_impl, v; kwargs..., __version=v"0.1.0")

hcat_impl(v; kwargs...) = reduce(hcat, v; kwargs...)
hcat_job(v; kwargs...) = create_job(hcat_impl, v; kwargs..., __version=v"0.1.0")



apply_impl(f, args...; kwargs...) = f(args...; kwargs...)
apply_job(f, args...; kwargs...) = create_job(apply_impl, f, args...; kwargs..., __version=v"0.1.0")

apply_broadcasted(f, args...; kwargs...) = f.(args...; kwargs...)
apply_broadcasted_job(f, args...; kwargs...) = create_job(apply_broadcasted, f, args...; kwargs..., __version=v"0.1.0")





function intersect_ind_impl(a, b)
	a == Colon() && return b
	b == Colon() && return a
	intersect(a,b)
end
function intersect_ind(::Preprocessing{E}, a, b) where E
	a == Colon() && return b
	b == Colon() && return a
	E && return create_job(Preprocess{false}(intersect_ind), a, b)
	return create_job(intersect_ind_impl, a, b; __version=v"0.1.0")
end

"""
	intersect_ind_job(a, b)

Create a spec to compute the intersection of `Vector`s `a` and `b` with indexes.
Just like `intersect`, but allows `a` and/or `b` to be `:`.
"""
intersect_ind_job(a, b) = create_job(Preprocess(intersect_ind), a, b)







function isequal_pre(::Preprocessing{E}, x, y) where E
	if isequal(x, y)
		true # early out
	elseif !(x isa SpecRef) && !(y isa SpecRef)
		false # early out
	elseif E
		create_job(Preprocess{false}(isequal_pre), x, y)
	else
		create_job(isequal, x, y; __version=v"0.1.0")
	end
end
isequal_job(x, y) = create_job(Preprocess(isequal_pre), x, y)


function indexin_impl(a::AbstractVector, b::AbstractVector; not_found)
	# TODO: Find better names `not_found`, `:error`, `:skip` and `:nothing`
	@assert not_found in (:error, :skip, :nothing)
	ind = indexin(a, b)

	if not_found == :error
		if any(isnothing, ind)
			d = string.(setdiff(a,b)[1:min(3,end)])
			error("Found $(count(isnothing,ind)) values in `a` (such as $d) that are not present in `b` (which contains e.g. $(b[1:min(3,end)])).")
		end
	elseif not_found == :skip
		all(isnothing, ind) && @warn "indexin_impl: No values in `a` (examples: $(a[1:min(3,end)])) were present in `b` (examples $(b[1:min(3,end)]))."
		filter!(!isnothing, ind)
	elseif not_found == :nothing && any(isnothing, ind)
		return ind
	end

	ind = convert(Vector{Int}, ind)
	SCPCore.simplify_ind(ind, length(b))
end

function indexin_impl(a::DataFrame, b::DataFrame; not_found)
	@assert ncol(a)==1
	@assert ncol(b)==1
	name_a = only(names(a,1))
	name_b = only(names(b,1))
	@assert name_a == name_b "Column names didn't match \"$name_a\" vs \"$name_b\"."
	indexin_impl(a[!,1], b[!,1]; not_found)
end

indexin_job(a, b; not_found=:error) = create_job(indexin_impl, a, b; not_found, __version=v"0.1.2")






nvar_job(data) = table_nrow_job(get_var_job(data))
Jobs.nvar(data) = nvar_job(data)

nobs_job(data) = table_nrow_job(get_obs_job(data))
Jobs.nobs(data) = nobs_job(data)


find_matching_ind_impl_job(f, df) = create_job(SCPCore.find_matching_ind, f, df; __version=v"0.1.4")


function find_matching_ind(action::Action, f, df; project_ids::Symbol)
	# @assert project_ids in (:no, :yes, :intersect)
	@assert project_ids in (:no, :yes, :intersect, :skip)
	if project_ids == :no
		f = action(f)
		df = action(df)
	end

	# TODO: If `f` is a pair, we can subset the columns of df to avoid involving them in the call
	if f === Colon()
		matching_ind = Colon()
	elseif f isa Pair
		k = first(f)

		# subset the columns to only depend on those that are used
		if k isa AbstractString
			x = get_columns_job(df, k)
			matching_ind = cached(find_matching_ind_impl_job(f, x))
		elseif k isa AbstractVector
			x = get_columns_job(df, k...)
			matching_ind = cached(find_matching_ind_impl_job(f, x))
		elseif k isa Union{SpecRef, DataFrame}
			# k is an "Annotation" - a DataFrame with an ID and a value column. Will be leftjoined and the function will be applied to the leftjoined vector with values.

			# TODO: Share code with `_extract_data_job`?
			ids_a = id_column_job(df)
			ids_b = id_column_job(k)
			ind_job = indexin_job(ids_a, ids_b; not_found=:nothing)
			v = value_column_data_job(k)
			x = getindex_or_missing_job(v, ind_job) # The values of the annotation `k`, reordered to match the order in df.

			matching_ind = cached(find_matching_ind_impl_job(last(f), x))
		else
			throw(ArgumentError("Unknown column selector $k of type $(typeof(k))."))
		end
	else
		# This is used when `f` is a function taking a DataFrameRow
		matching_ind = cached(find_matching_ind_impl_job(f, df))
	end

	if action isa Eval || project_ids == :no
		return matching_ind
	elseif project_ids == :skip # Experimental - is this a good name?
		return Colon()
	else
		# We need to remap the indices, going through IDs
		ids = id_column_job(df)
		ids2 = action(ids) # IDs from projected dataset

		cond = isequal_job(ids, ids2)

		# matching_ids = table_getindex_impl_job(ids, matching_ind) # unprojected IDs (NB: this will simplify if matching_ind==Colon())
		matching_ids = table_getindex_job(ids, matching_ind) # unprojected IDs (NB: this will simplify if matching_ind==Colon())
		if project_ids == :yes
			proj_ind = indexin_job(matching_ids, ids2; not_found=:error) # Use order from unprojected
		else#if project_ids == :intersect
			proj_ind = indexin_job(matching_ids, ids2; not_found=:skip) # Use order from unprojected
		end

		# This gives us an early out when ids==ids2, since we can just return matching_ind in that case (no need to bother with getting matching ids and doing indexin)
		ifelse_job(cond, matching_ind, proj_ind)
	end
end
create_find_matching_ind_job(f, df; project_ids) =
	create_job(Projectable(find_matching_ind), f, df; project_ids)
# Jobs.find_matching_ind(args...; kwargs...) =
# 	create_find_matching_ind_job(args...; kwargs...)




_nrow(df::DataFrame) = nrow(df)
_nrow(v::AbstractVector) = length(df)
function _colon_to_single_ind(x)
	n = _nrow(x)
	n==1 ? 1 : error("Expected a single element, got $n.")
end
_colon_to_single_ind_job(x) = create_job(_colon_to_single_ind, x; __version=v"0.1.0")

function find_single_ind(::Preprocessing, f, df; project_id::Symbol)
	ind = create_find_matching_ind_job(f, df; project_ids=project_id)

	cond = isequal_job(ind, Colon())
	a = _colon_to_single_ind_job(df) # This is an obscure edge case, because `:` is allowed iff there is only one element in the container. We could handle it nicer with more preprocessing, but it's probably not worth it.
	b = apply_job(only, ind)
	ifelse_pr_job(cond, a, b)
end

find_single_ind_job(f, df; project_id) =
	create_job(Preprocess(find_single_ind), f, df; project_id)












matrix_getindex_impl(matrix; kwargs...) =
	create_job(SCPCore.matrix_getindex, matrix; kwargs..., __version=v"0.1.0")


function compose_ind(inner::Union{Colon, AbstractVector{<:Integer}}, outer::Union{Colon, AbstractVector{<:Integer}})
	inner === Colon() && return outer
	outer === Colon() && return inner
	inner[outer]
end

function matrix_getindex_pre(::Preprocessing{false}, matrix; var_ind, obs_ind)
	if var_ind === Colon() && obs_ind === Colon()
		matrix
	elseif is_hblock(matrix)
		if obs_ind === Colon()
			hblock_map(matrix) do x
				matrix_getindex_impl(x; var_ind, obs_ind)
			end
		else
			# We need to update obs_ind to match each the range of each block
			blocks = matrix.args[1]
			ranges = _get_kwarg(matrix, :ranges)
			@assert length(blocks) == length(ranges)

			new_obs_ind, new_ranges = SCPCore.ind_to_blocked_ind(obs_ind, ranges)
			new_blocks = [matrix_getindex_pre_job(b; var_ind, obs_ind=I) for (b,I) in zip(blocks, new_obs_ind)]
			hblock_job(new_blocks, new_ranges)
		end
	elseif matrix.f === SCPCore.matrix_getindex
		# Collapse nested getindex calls which is important for getting canonical representations
		inner_matrix  = matrix.args[1]
		inner_var_ind = _get_kwarg(matrix, :var_ind)
		inner_obs_ind = _get_kwarg(matrix, :obs_ind)

		composed_var = compose_ind(inner_var_ind, var_ind)
		composed_obs = compose_ind(inner_obs_ind, obs_ind)

		matrix_getindex_impl(inner_matrix; var_ind=composed_var, obs_ind=composed_obs)
	else
		matrix_getindex_impl(matrix; var_ind, obs_ind)
	end
end

matrix_getindex_pre_job(matrix; var_ind, obs_ind) =
	create_job(Preprocess{false}(matrix_getindex_pre), matrix; var_ind, obs_ind)



function _matrix_ind_job(action::Action, ind)
	ind === nothing && return Colon()
	ind_p = action(ind)
	if action isa Projection && !(ind isa SpecRef)
		cond = isequal_job(ind, ind_p)
		ind_p = ifelse_job(cond, ind_p, _getindex_error_job(ind))
	end
	return fetched(ind_p)
end

function matrix_getindex_pr(action::Action, matrix; var_ind=nothing, obs_ind=nothing)
	matrix = action(matrix)
	var_ind = _matrix_ind_job(action, var_ind)
	obs_ind = _matrix_ind_job(action, obs_ind)
	matrix_getindex_pre_job(matrix; var_ind, obs_ind)
end

function create_matrix_getindex_job(matrix; kwargs...)
	create_job(Projectable(matrix_getindex_pr), matrix; kwargs...)
end



datamatrix_getindex(::Mat, data; kwargs...) =
	create_matrix_getindex_job(get_matrix_job(data); kwargs...)
function datamatrix_getindex(::Var, data; var_ind=nothing, kwargs...)
	var = get_var_job(data)
	var_ind === nothing ? var : table_getindex_job(var, var_ind)
end
function datamatrix_getindex(::Obs, data; obs_ind=nothing, kwargs...)
	obs = get_obs_job(data)
	obs_ind === nothing ? obs : table_getindex_job(obs, obs_ind)
end


create_datamatrix_getindex_job(data; kwargs...) =
	create_job(DataMatrixFunction(datamatrix_getindex), data; kwargs...)



# These are temporary workarounds. Will be solved later when block refactoring is complete.
function _maybe_col_blockify(S, A::SCPCore.Blocks)
	row_ranges = SCPCore.get_row_ranges(A)
	if length(row_ranges)>1
		SCPCore.blockify(S; row_ranges=(Colon(),), col_ranges=row_ranges)
	else
		S
	end
end
_maybe_col_blockify(S, A::SCPCore.MatrixExpressions.MatrixSum) = _maybe_col_blockify(S, A.terms[1].matrix) # Assuming the first term is the blocked sparse matrix as per sctransform/logtransform

function _maybe_row_blockify(S, A::SCPCore.Blocks)
	col_ranges = SCPCore.get_col_ranges(A)
	if length(col_ranges)>1
		SCPCore.blockify(S; row_ranges=col_ranges, col_ranges=(Colon(),))
	else
		S
	end
end
_maybe_row_blockify(S, A::SCPCore.MatrixExpressions.MatrixSum) = _maybe_row_blockify(S, A.terms[1].matrix) # Assuming the first term is the blocked sparse matrix as per sctransform/logtransform




function get_matrix_row(matrix, ind::Integer)
	P,N = size(matrix)
	@assert 1 <= ind <= P
	if matrix isa AbstractMatrix
		res = matrix[ind, :]
	else # Matrix Expression or Blocks
		S = sparse([1], [ind], true, 1, P) # We might need to block it to match `matrix`.
		S = _maybe_col_blockify(S, matrix)
		res = S*matrix
		@assert size(res) == (1,N)
		res = convert(Matrix, res) # needed for blocks
		res = vec(res)
	end
	convert(Vector, res) # ensure it's dense
end
function get_matrix_col(matrix, ind::Integer)
	P,N = size(matrix)
	@assert 1 <= ind <= N
	if matrix isa AbstractMatrix
		res = matrix[:, ind]
	else # Matrix Expression or Blocks
		S = sparse([ind], [1], true, N, 1) # We might need to block it to match `matrix`.
		S = _maybe_row_blockify(S, matrix)
		res = matrix*S
		@assert size(res) == (P,1)
		res = convert(Matrix, res) # needed for blocks
		res = vec(res)
	end
	convert(Vector, res) # ensure it's dense
end
get_matrix_row_job(matrix, ind) =
	create_job(get_matrix_row, matrix, ind; __version=v"0.1.0")
get_matrix_col_job(matrix, ind) =
	create_job(get_matrix_col, matrix, ind; __version=v"0.1.0")





prefixed_id_values(prefix::String, n) = string.(prefix, 1:n)
function prefixed_ids(::Preprocessing, col::String, prefix, n)
	col_data = create_job(prefixed_id_values, prefix, n; __version=v"0.1.0")
	create_table_job(col=>col_data)
end
prefixed_ids_job(col, prefix, n) =
	create_job(Preprocess(prefixed_ids), col, prefix, n)
