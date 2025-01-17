# TODO: Replace filter.jl with this file.

find_matching_ids(f, df::DataFrame) =
	select(filter(f, df; view=true), 1)

find_matching_ids(::Colon, df::DataFrame) =
	select(df, 1; copycols=false) # input is considered read-only, so we don't need to copy


function ids_to_indices(df::DataFrame, ids::DataFrame)
	@assert size(ids, 2)==1
	@assert names(df, 1) == names(ids, 1)

	ind = indexin(ids[!,1], df[!,1])
	@assert all(!isnothing, ind) # TODO: Add boolean parameter allowing that only a subset is present
	something.(ind) # remove `Nothing` from eltype (and error if `nothing` is encountered)
end
ids_to_indices(df::DataFrame, ::Colon) = 1:size(df,1)


# The name is chosen since it is akin to getindex.
function annotation_getindex(df::DataFrame, ind::Vector{Int})
	if index_isnoop(ind,size(df,1))
		df # input is considered read-only, so we don't need to copy
	else
		df[ind, :]
	end
end

# The name is chosen since it is akin to getindex
function matrix_getindex(matrix; var_ind::Union{Vector{Int},Colon}=:, obs_ind::Union{Vector{Int},Colon}=:)
	if index_isnoop(var_ind, size(matrix,1)) && index_isnoop(obs_ind, size(matrix,2))
		matrix # input is considered read-only, so we don't need to copy
	else
		_subsetmatrix(matrix, var_ind, obs_ind)
	end
end
