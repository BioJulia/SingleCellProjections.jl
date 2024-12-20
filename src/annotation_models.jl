struct AnnotationVectorModel <: ProjectionModel end

# No. This reordering should be a separate step to make projections work well.
function _value_vector_reorder(obs_id::DataFrame, v)
	df = _get_table(v)
	@assert size(obs_id,2) == 1
	@assert size(df,2) == 2
	@assert names(obs_id,1) == names(df,1)

	ind = indexin(obs_id[!,1], df[!,1])
	if any(isnothing, ind)
		n = count(isnothing, ind)
		i = findfirst(isnothing,ind)
		throw(ArgumentError("Error creating ValueVector, found $n missing IDs (first missing ID=\"$(obs_id[i,1]))\"."))
	end

	ind = identity.(ind) # get rid of Nothings
	df[ind,2]
end


function project2(::AnnotationVectorModel, id::DataFrame, v)
	df = _get_table(v)

	@assert size(obs_id,2) == 1
	@assert names(obs_id,1) == names(df,1)
	@assert size(df,2) == 2 # Do not allow multiple annotation columns, because then we couldn't return a Vector, and a Matrix doesn't work either because the eltype would be weird.

	ind = indexin(obs_id[!,1], df[!,1])
	if any(isnothing, ind)
		n = count(isnothing, ind)
		i = findfirst(isnothing,ind)
		throw(ArgumentError("Error subsetting annotation, $n/$(size(df,1)) IDs are missing (first missing ID=\"$(obs_id[i,1]))\"."))
	end

	ind = identity.(ind) # get rid of Nothings
	df[ind,2]
end
