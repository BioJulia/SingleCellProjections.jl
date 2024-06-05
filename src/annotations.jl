struct Annotations
	df::DataFrame # implementation detail, might be changed later. The first column is the ID column, and the name of that column is the name of the axis.
end

function Base.getproperty(a::Annotations, column::Symbol)
	df = getfield(a,:df)
	column = String(column)
	id_column = names(df, 1)
	cols = only(id_column) == column ? id_column : vcat(id_column,column)
	Annotations(select(df, cols))
end

# function annotation_id(a::Annotations)
# 	df = getfield(a,:df)
# 	only(names(df,1))
# end
function annotation_name(a::Annotations)
	df = getfield(a,:df)
	@assert size(df,2) == 2 "Expected annotations to object to have an ID column and a single data column, got columns: $(names(df))"
	only(names(df,2))
end

function annotation_values(a::Annotations)
	df = getfield(a,:df)
	@assert size(df,2) == 2 "Expected annotations to object to have an ID column and a single data column, got columns: $(names(df))"
	df[!,2]
end
