table_validatecols(table, col) =
	hasproperty(table, Symbol(col)) || throw(ArgumentError("ID column $col not found in table with columns $(join(names(table)))."))

function table_validateunique(table, col)
	bad_ind = findfirst(nonunique(table, col))
	bad_ind !== nothing && error("ID \"", table[bad_ind,col], "\" is not unique.")
end

table_cols_equal(a, b; cols=names(b)) =
	isequal(select(a, cols; copycols=false), select(b, cols; copycols=false))

function table_indexin(a, b; cols=names(b), kwargs...)
	b = select(b, cols; copycols=false)
	a = select(a, cols; copycols=false)
	b.__index__ .= 1:size(b,1)
	leftjoin!(a,b; on=cols, kwargs...)
	coalesce.(a.__index__, nothing)
end
