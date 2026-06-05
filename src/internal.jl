# WIP - some of these might not be needed.

# This must be used instead of ifelse_spec when `cond` should be projected. Otherwise it will be fetched **before** projections take place.
ifelse_pr(action::Action, cond, x, y) = ifelse_spec(action(cond), action(x), action(y))
ifelse_pr_spec(cond, x, y) = create_spec(Projectable(ifelse_pr), cond, x, y)


# Find a better name?
function combine_vectors_impl(args...; delim=nothing)
	if delim !== nothing
		n = length(args)
		args = Iterators.take(Iterators.flatten(zip(args, Iterators.cycle(delim))), 2n-1) # insert delim between
	end
	string.(args...)
end
combine_vectors_spec(args...; kwargs...) = create_spec(combine_vectors_impl, args...; kwargs..., __version=v"0.1.0")





_getindex_error(ind) = throw(ArgumentError("Raw indices not allowed when projecting (unless containers are identical). Got indices: $ind."))
_getindex_error_spec(ind) = create_spec(_getindex_error, ind; __version=v"0.1.0")

# getindex_impl(::Preprocessing, v, ind) = ind===Colon() ? v : create_spec(getindex, v, ind; __version=v"0.1.0")
# function getindex_impl(::Preprocessing{E}, v, ind) where E
# 	if ind === Colon()
# 		v
# 	elseif E
# 		create_spec(Preprocess{false}(getindex_impl), v, fetched(ind)) # NB: This way we fetch after projections are handled!
# 	else
# 		create_spec(getindex, v, ind; __version=v"0.1.0")
# 	end
# end

function getindex_impl(::Preprocessing, v, ind)
	if ind === Colon()
		v # Projections have been handled, so indexing by `:` is OK
	elseif v isa SpecRef && v.f === getindex
		# Collapse nested getindex calls which is important for getting canonical representations
		create_spec(getindex, v.args[1], compose_ind(v.args[2], ind); __version=v"0.1.0")
	else
		create_spec(getindex, v, ind; __version=v"0.1.0")
	end
end
getindex_impl_spec(v, ind) = create_spec(Preprocess{false}(getindex_impl), v, fetched(ind))

function getindex_pr(action, v, ind)
	v_p = action(v)
	result = getindex_impl_spec(v_p, action(ind))

	if action isa Projection && !(ind isa SpecRef) # TODO: Fix, this will trigger even if ind is replaced by the action, which it shouldn't - maybe hard to avoid?
		cond = isequal_spec(v, v_p)
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

	if action isa Projection && !(ind isa SpecRef)
		cond = isequal_spec(v, v_p)
		result = ifelse_spec(cond, result, _getindex_or_missing_error_spec(ind)) # TODO: Fix _getindex_or_missing_error_spec isn't defined anywhere...
	end

	result
end
getindex_or_missing_spec(v, ind) = create_spec(Projectable(getindex_or_missing_pr), v, ind)



intersect_spec(a, b, args...) = create_spec(intersect, a, b, args...; __version=v"0.1.0")
length_spec(x) = create_spec(length, x; __version=v"0.1.0")
unique_spec(x) = create_spec(unique, x; __version=v"0.1.0")
join_spec(x, args...) = create_spec(join, x, args...; __version=v"0.1.0")
reshape_spec(A, args...) = create_spec(reshape, A, args...; __version=v"0.1.0")
repeat_spec(A, args...; kwargs...) = create_spec(repeat, A, args...; kwargs..., __version=v"0.1.0")
prod_spec(args...; kwargs...) = create_spec(prod, args...; kwargs..., __version=v"0.1.0")
allequal_spec(x) = create_spec(allequal, x; __version=v"0.1.0")

# vcat_spec(args...; kwargs...) = create_spec(vcat, args...; kwargs..., __version=v"0.1.0")
# hcat_spec(args...; kwargs...) = create_spec(hcat, args...; kwargs..., __version=v"0.1.0")

vcat_impl(v; kwargs...) = reduce(vcat, v; kwargs...)
vcat_spec(v; kwargs...) = create_spec(vcat_impl, v; kwargs..., __version=v"0.1.0")

hcat_impl(v; kwargs...) = reduce(hcat, v; kwargs...)
hcat_spec(v; kwargs...) = create_spec(hcat_impl, v; kwargs..., __version=v"0.1.0")



apply_impl(f, args...; kwargs...) = f(args...; kwargs...)
apply_spec(f, args...; kwargs...) = create_spec(apply_impl, f, args...; kwargs..., __version=v"0.1.0")

apply_broadcasted(f, args...; kwargs...) = f.(args...; kwargs...)
apply_broadcasted_spec(f, args...; kwargs...) = create_spec(apply_broadcasted, f, args...; kwargs..., __version=v"0.1.0")





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
	elseif !(x isa SpecRef) && !(y isa SpecRef)
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
	if ind == 1:length(b)
		return Colon() # simplify common case
	else
		return ind
	end
end

function indexin_impl(a::DataFrame, b::DataFrame; not_found)
	@assert ncol(a)==1
	@assert ncol(b)==1
	name_a = only(names(a,1))
	name_b = only(names(b,1))
	@assert name_a == name_b "Column names didn't match \"$name_a\" vs \"$name_b\"."
	indexin_impl(a[!,1], b[!,1]; not_found)
end

indexin_spec(a, b; not_found=:error) = create_spec(indexin_impl, a, b; not_found, __version=v"0.1.2")






nvar_spec(data) = table_nrow_spec(get_var_spec(data))
Jobs.nvar(data) = nvar_spec(data)

nobs_spec(data) = table_nrow_spec(get_obs_spec(data))
Jobs.nobs(data) = nobs_spec(data)


find_matching_ind_impl_spec(f, df) = create_spec(SCPCore.find_matching_ind, f, df; __version=v"0.1.4")


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
		elseif k isa Union{SpecRef, DataFrame}
			# k is an "Annotation" - a DataFrame with an ID and a value column. Will be leftjoined and the function will be applied to the leftjoined vector with values.

			# TODO: Share code with `_extract_data_spec`?
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
	elseif project_ids == :skip # Experimental - is this a good name?
		return Colon()
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
# 	create_find_matching_ind_spec(args...; kwargs...)




_nrow(df::DataFrame) = nrow(df)
_nrow(v::AbstractVector) = length(df)
function _colon_to_single_ind(x)
	n = _nrow(x)
	n==1 ? 1 : error("Expected a single element, got $n.")
end
_colon_to_single_ind_spec(x) = create_spec(_colon_to_single_ind, x; __version=v"0.1.0")

function find_single_ind(::Preprocessing, f, df; project_id::Symbol)
	ind = create_find_matching_ind_spec(f, df; project_ids=project_id)

	cond = isequal_spec(ind, Colon())
	a = _colon_to_single_ind_spec(df) # This is an obscure edge case, because `:` is allowed iff there is only one element in the container. We could handle it nicer with more preprocessing, but it's probably not worth it.
	b = apply_spec(only, ind)
	ifelse_pr_spec(cond, a, b)
end

find_single_ind_spec(f, df; project_id) =
	create_spec(Preprocess(find_single_ind), f, df; project_id)




# Do we need these, or upstream steps always return Colon() when they can?
index_isnoop_spec(ind, n) =
	create_spec(SCPCore.index_isnoop, ind, n; __version=v"0.0.1")
simplify_ind_spec(ind, n) =
	ind === Colon() ? Colon() : ifelse_spec(index_isnoop_spec(ind, n), Colon(), ind) # early out if is already known to be Colon








matrix_getindex_impl(matrix; kwargs...) =
	create_spec(SCPCore.matrix_getindex, matrix; kwargs..., __version=v"0.1.0")

# # TODO: split this by hblock
# function matrix_getindex_pre(::Preprocessing, matrix; var_ind, obs_ind)
# 	if var_ind == Colon() && obs_ind == Colon()
# 		matrix
# 	else
# 		matrix_getindex_impl(matrix; var_ind, obs_ind)
# 	end
# end

compose_ind(inner::Union{Colon, AbstractVector{<:Integer}}, outer::Union{Colon, AbstractVector{<:Integer}}) =
	inner === Colon() ? outer :
	outer === Colon() ? inner :
	inner[outer]

function matrix_getindex_pre(::Preprocessing, matrix; var_ind, obs_ind)
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

			# first_ind = searchsortedfirst.(Ref(obs_ind), first.(ranges))
			# last_ind = vcat(@view(first_ind[2:end]).-1, length(obs_ind))

			# new_ranges = range.(first_ind, last_ind)
			# new_blocks = map(1:length(blocks)) do i
			# 	new_obs_ind = obs_ind[new_ranges[i]] .- first(ranges[i]) .+ 1
			# 	new_obs_ind = simplify_ind_spec(new_obs_ind, length(ranges[i])) # we need to simplify again, because we might get Colon() for some blocks but not others
			# 	matrix_getindex_pre_spec(blocks[i]; var_ind, obs_ind=new_obs_ind)
			# end

			new_obs_ind, new_ranges = SCPCore.ind_to_blocked_ind(obs_ind, ranges)
			new_obs_ind = simplify_ind_spec.(new_obs_ind, length.(ranges))
			new_blocks = [matrix_getindex_pre_spec(b; var_ind, obs_ind=I) for (b,I) in zip(blocks, new_obs_ind)]
			hblock_spec(new_blocks, new_ranges)
		end
	elseif matrix.f === SCPCore.matrix_getindex
		# Collapse nested getindex calls which is important for getting canonical representations
		inner_matrix  = matrix.args[1]
		inner_var_ind = _get_kwarg(matrix, :var_ind)
		inner_obs_ind = _get_kwarg(matrix, :obs_ind)
		matrix_getindex_pre_spec(inner_matrix;
			var_ind = compose_ind(inner_var_ind, var_ind),
			obs_ind = compose_ind(inner_obs_ind, obs_ind))
	else
		matrix_getindex_impl(matrix; var_ind, obs_ind)
	end
end

matrix_getindex_pre_spec(matrix; var_ind, obs_ind) =
	create_spec(Preprocess{false}(matrix_getindex_pre), matrix; var_ind, obs_ind)



function _matrix_ind_spec(action::Action, ind, n=nothing)
	ind === nothing && return Colon()

	ind_p = action(ind)
	if action isa Projection && !(ind isa SpecRef)
		cond = isequal_spec(ind, ind_p)
		ind_p = ifelse_spec(cond, ind_p, _getindex_error_spec(ind))
	end
	# TODO: Where to simplify_ind ?
	n !== nothing && (ind_p = simplify_ind_spec(ind_p, n))
	return fetched(ind_p)
end

function matrix_getindex_pr(action::Action, matrix; var_ind=nothing, obs_ind=nothing, nvar=nothing, nobs=nothing)
	matrix = action(matrix)
	var_ind = _matrix_ind_spec(action, var_ind, nvar)
	obs_ind = _matrix_ind_spec(action, obs_ind, nobs)
	# create_spec(Preprocess{false}(matrix_getindex_pre), matrix; var_ind, obs_ind)
	matrix_getindex_pre_spec(matrix; var_ind, obs_ind)
end

function create_matrix_getindex_spec(matrix; kwargs...)
	create_spec(Projectable(matrix_getindex_pr), matrix; kwargs...)
end



datamatrix_getindex(::Mat, data; kwargs...) =
	create_matrix_getindex_spec(get_matrix_spec(data); nvar=nvar_spec(data), nobs=nobs_spec(data), kwargs...)
function datamatrix_getindex(::Var, data; var_ind=nothing, kwargs...)
	var = get_var_spec(data)
	var_ind === nothing ? var : table_getindex_spec(var, var_ind)
end
function datamatrix_getindex(::Obs, data; obs_ind=nothing, kwargs...)
	obs = get_obs_spec(data)
	obs_ind === nothing ? obs : table_getindex_spec(obs, obs_ind)
end


create_datamatrix_getindex_spec(data; kwargs...) =
	create_spec(DataMatrixFunction(datamatrix_getindex), data; kwargs...)



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
get_matrix_row_spec(matrix, ind) =
	create_spec(get_matrix_row, matrix, ind; __version=v"0.1.0")
get_matrix_col_spec(matrix, ind) =
	create_spec(get_matrix_col, matrix, ind; __version=v"0.1.0")





prefixed_id_values(prefix::String, n) = string.(prefix, 1:n)
function prefixed_ids(::Preprocessing, col::String, prefix, n)
	col_data = create_spec(prefixed_id_values, prefix, n; __version=v"0.1.0")
	create_table_spec(col=>col_data)
end
prefixed_ids_spec(col, prefix, n) =
	create_spec(Preprocess(prefixed_ids), col, prefix, n)
