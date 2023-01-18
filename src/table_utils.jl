table_hascols(table, cols) = all(x->hasproperty(table,x), cols)

table_validatecols(table, cols) =
	table_hascols(table, cols) || throw(ArgumentError("ID columns $cols not found in table with columns $(join(names(table)))."))

function table_validateunique(table, cols)
	bad_ind = findfirst(nonunique(table, cols))
	bad_ind !== nothing && error("ID [", join(table[bad_ind,cols],", "), "] is not unique.")
end

table_cols_equal(a, b; cols=names(b)) =
	isequal(select(a, cols; copycols=false), select(b, cols; copycols=false))

function table_indexin(a, b; cols=names(b))
	b = select(b, cols; copycols=false)
	a = select(a, cols; copycols=false)
	b.__index__ .= 1:size(b,1)
	leftjoin!(a,b; on=cols)
	coalesce.(a.__index__, nothing)
end
