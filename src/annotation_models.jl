extract_annotation(annotation::DataFrame, name::String) =
	annotation[!,name]



# Old, but we might want something similar later to handle external annotations
# function extract_annotation(id::DataFrame, v)
# 	df = _get_table(v)

# 	@assert size(id,2) == 1
# 	@assert names(id,1) == names(df,1)
# 	@assert size(df,2) == 2 # Do not allow multiple annotation columns, because then we couldn't return a Vector, and a Matrix doesn't work either because the eltype would be weird.

# 	ind = indexin(id[!,1], df[!,1])
# 	if any(isnothing, ind)
# 		n = count(isnothing, ind)
# 		i = findfirst(isnothing,ind)
# 		throw(ArgumentError("Error subsetting annotation, $n/$(size(df,1)) IDs are missing (first missing ID=\"$(id[i,1]))\"."))
# 	end

# 	ind = something.(ind) # remove `Nothing` from eltype (and error if `nothing` is encountered)
# 	df[ind,2]
# end

# And use StatelessModel{extract_annotation}

# struct ExtractAnnotationModel <: ProjectionModel end

# function project2(::ExtractAnnotationModel, id::DataFrame, v)
# 	df = _get_table(v)

# 	@assert size(id,2) == 1
# 	@assert names(id,1) == names(df,1)
# 	@assert size(df,2) == 2 # Do not allow multiple annotation columns, because then we couldn't return a Vector, and a Matrix doesn't work either because the eltype would be weird.

# 	ind = indexin(id[!,1], df[!,1])
# 	if any(isnothing, ind)
# 		n = count(isnothing, ind)
# 		i = findfirst(isnothing,ind)
# 		throw(ArgumentError("Error subsetting annotation, $n/$(size(df,1)) IDs are missing (first missing ID=\"$(id[i,1]))\"."))
# 	end

# 	ind = something.(ind) # remove `Nothing` from eltype (and error if `nothing` is encountered)
# 	df[ind,2]
# end
