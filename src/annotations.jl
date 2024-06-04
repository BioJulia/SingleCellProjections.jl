struct Annotations
	df::DataFrame # implementation detail, might be changed later. The first column is the ID column, and the name of that column is the name of the axis.
end

function Base.getproperty(a::Annotations, column::Symbol)
	df = getfield(a,:df)
	column = String(column)
	id_column = names(df, 1)
	cols = only(id_column) == column ? id_column : vcat(id_column,column)
	select(df, cols)
end
