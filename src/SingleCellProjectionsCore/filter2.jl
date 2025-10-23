# TODO: Replace filter.jl with this file.

find_matching_ids(f, df::DataFrame) =
	select(filter(f, df; view=true), 1)

find_matching_ids(::Colon, df::DataFrame) =
	select(df, 1; copycols=false) # input is considered read-only, so we don't need to copy


function ids_to_indices(df::DataFrame, ids::DataFrame)
	@assert size(ids, 2)==1
	@assert names(df, 1) == names(ids, 1)

	ind = indexin(ids[!,1], df[!,1])
	# @assert all(!isnothing, ind)
	if any(isnothing,ind)
		missing_ids = ids[ind .== nothing, 1]
		n_missing = length(missing_ids)
		n_total = size(ids,1)
		msg = string("$n_missing/$n_total IDs where not found: ", join(missing_ids[1:10], ','), n_missing>10 ? "..." : "")
		throw(ArgumentError(msg))
	end

	ind = something.(ind) # remove `Nothing` from eltype (and error if `nothing` is encountered)

	if ind == 1:size(df,1)
		return Colon() # Simplify common case
	else
		return ind
	end
end
# ids_to_indices(df::DataFrame, ::Colon) = 1:size(df,1)
ids_to_indices(df::DataFrame, ::Colon) = Colon()


# The name is chosen since it is akin to getindex.
function annotation_getindex(df::DataFrame, ind::Union{<:AbstractVector{Int},Colon})
	if index_isnoop(ind,size(df,1))
		df # input is considered read-only, so we don't need to copy
	else
		df[ind, :]
	end
end

# The name is chosen since it is akin to getindex
function matrix_getindex(matrix; var_ind::Union{<:AbstractVector{Int},Colon}=:, obs_ind::Union{<:AbstractVector{Int},Colon}=:)
	if index_isnoop(var_ind, size(matrix,1)) && index_isnoop(obs_ind, size(matrix,2))
		matrix # input is considered read-only, so we don't need to copy
	else
		_subsetmatrix(matrix, var_ind, obs_ind)
	end
end
