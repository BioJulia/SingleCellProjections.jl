# NB: Column names here are fixed and expected to be strings.
create_table_impl(args::Pair{String,<:Any}...) = DataFrame(args...; copycols=false)
create_table_impl_spec(args...) = create_spec(create_table_impl, args...; __version=v"0.1.0")

create_table_pr(action::Action, args::Pair{String,<:Any}...) = create_table_impl_spec(action(args)...)
create_table_spec(args...) = create_spec(Projectable(create_table_pr), args...)


is_create_table_spec(x::Spec) = x.f in (create_table_pr, create_table_impl)
is_create_table_spec(::Any) = false

forwarded_to_table(spec) = forwarded(is_create_table_spec, spec)



function table_from_compound_result(compound_result, colnames)
	cols = (name=>cached(compound_result, name) for name in colnames)
	create_table_impl_spec(cols...)
end
function table_from_compound_result(compound_result)
	colnames = fetched(cached(compound_result; return_keys=true))
	create_spec(Preprocess(table_from_compound_result), compound_result, colnames)
end



get_colnames_impl(table) = names(table)
function get_colnames_pre(table)
	if is_create_table_spec(table)
		return first.(table.args)
	else
		return create_spec(get_colnames_impl, table; __version=v"0.1.0")
	end
end
get_colnames_pr(action, table) =
	create_spec(Preprocess(get_colnames_pre), forwarded_to_table(action(table)))
get_colnames(table_spec) = create_spec(Projectable(get_colnames_pr), table_spec)
get_colnames_spec(table) = create_spec(Preprocess(get_colnames), table)
Jobs.get_colnames(table) = Job(get_colnames_spec(table))



get_id_colname_impl(table) = only(names(table,1))
function get_id_colname_pre(table)
	if is_create_table_spec(table)
		return first(first(table.args)) # key of first column
	else
		return create_spec(get_id_colname_impl, table; __version=v"0.1.0")
	end
end
get_id_colname_pr(action, table) =
	create_spec(Preprocess(get_id_colname_pre), forwarded_to_table(action(table)))
get_id_colname(table_spec) = create_spec(Projectable(get_id_colname_pr), table_spec)
get_id_colname_spec(table) = create_spec(Preprocess(get_id_colname), table)
Jobs.get_id_colname(table) = Job(get_id_colname_spec(table))


function get_value_colname_impl(table)
	len = ncol(table)
	len == 2 || throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
	only(names(table,2))
end
function get_value_colname_pre(table)
	if is_create_table_spec(table)
		len = length(table.args)
		len == 2 || throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
		return first(table.args[2]) # key of second column
	else
		return create_spec(get_value_colname_impl, table; __version=v"0.1.0")
	end
end
get_value_colname_pr(action, table) =
	create_spec(Preprocess(get_value_colname_pre), forwarded_to_table(action(table)))
get_value_colname(table_spec) = create_spec(Projectable(get_value_colname_pr), table_spec)
get_value_colname_spec(table) = create_spec(Preprocess(get_value_colname), table)
Jobs.get_value_colname(table) = Job(get_value_colname_spec(table))



get_columns_impl(table, colnames::String...) = select(table, [colnames...])
function get_columns_pre(table, colnames::String...)
	if is_create_table_spec(table)
		table_colnames = first.(table.args)
		ind = indexin(colnames, table_colnames)
		any(isnothing, ind) && throw(ArgumentError("The following column names where not found: $(setdiff(colnames, table_colnames))"))
		create_table_impl_spec(table.args[ind]...)
	else
		return create_spec(get_columns_impl, table, colnames...; __version=v"0.1.0")
	end
end
get_columns_pr(action, table, colnames::String...) =
	create_spec(Preprocess(get_columns_pre), forwarded_to_table(action(table)), colnames...)
get_columns(table_spec, colnames...) = create_spec(Projectable(get_columns_pr), table_spec, colnames...)
get_columns_spec(table, colname1, colnames...) = create_spec(Preprocess(get_columns), table, colname1, colnames...)
Jobs.get_columns(table, colname1, colnames...) = Job(get_columns_spec(table, colname1, colnames...))



id_column(table) = get_columns_spec(table, fetched(get_id_colname_spec(table))) # TODO: Use 1 as index instead?
id_column_spec(table) = create_spec(Preprocess(id_column), table)
Jobs.id_column(table) = Job(id_column_spec(table))

annotation(table, colname) = get_columns_spec(table, fetched(get_id_colname_spec(table)), colname)
annotation_spec(table, colname) = create_spec(Preprocess(annotation), table, colname)
Jobs.annotation(table, colname) = Job(annotation_spec(table, colname))



_col_ind(::Any, col::Integer) = col
function _col_ind(table, col::String)
	i = findfirst(p->isequal(p[1], col), table.args)
	i === nothing && throw(KeyError(col))
	i
end

column_data_impl(table, col) = table[!,col]
function column_data_pre(table, col)
	if is_create_table_spec(table)
		i = _col_ind(table, col)
		return table.args[i][2]
	else
		return create_spec(column_data_impl, table, col; __version=v"0.1.0")
	end
end
column_data_pr(action, table, col) =
	create_spec(Preprocess(column_data_pre), forwarded_to_table(action(table)), col)
column_data(table, col) = create_spec(Projectable(column_data_pr), table, col)
column_data_spec(table, col) = create_spec(Preprocess(column_data), table, col)
Jobs.column_data(table, col) = Job(column_data_spec(table, col))




table_nrow(table) = length_spec(column_data_spec(table,1))
table_nrow_spec(table) = create_spec(Preprocess(table_nrow), table)
Jobs.table_nrow(table) = Job(table_nrow_spec(table))



_add_column_length_error(n1, n2, name) = throw(ArgumentError("Expected column \"$name\" to have length $n1, but got $n2."))
_add_column_length_error_spec(n1, n2, name) = create_spec(_add_column_length_error, n1, n2, name)

add_column_impl(table, name, column) = hcat(table, DataFrame(name=>column; copycols=false); copycols=false)
function add_column_pre(table, name, column)
	if is_create_table_spec(table)
		# Check that there is no column with that name
		if name in (k for (k,_) in table.args)
			throw(ArgumentError("A column with the name \"$name\" already exists."))
		end

		result = create_table_impl_spec(table.args..., name=>column)

		# Check that the length of the new column matches the old
		n1 = table_nrow_spec(table)
		n2 = length_spec(column)
		cond = isequal_spec(n1, n2)
		return ifelse_spec(cond, result, _add_column_length_error_spec(n1,n2,name))
	else
		return create_spec(add_column_impl, table, name, column; __version=v"0.1.0")
	end
end
add_column_pr(action, table, name, column) =
	create_spec(Preprocess(add_column_pre), forwarded_to_table(action(table)), name, action(column))
add_column(table, name, column) = create_spec(Projectable(add_column_pr), table, name, column)
add_column_spec(table, name, column) = create_spec(Preprocess(add_column), table, name, column)
Jobs.add_column(table, name, column) = Job(add_column_spec(table, name, column))



table_getindex_impl(table, ind) = table[ind,:]
function table_getindex_pre(table, ind)
	if is_create_table_spec(table)
		cols = (k=>getindex_impl_spec(v, ind) for (k,v) in table.args)
		return create_table_impl_spec(cols...)
	else
		return create_spec(table_getindex_impl, table, ind; __version=v"0.1.0")
	end
end
table_getindex_pr(action, table, ind) =
	create_spec(Preprocess(table_getindex_pre), forwarded_to_table(action(table)), action(ind))
table_getindex(table, ind) = create_spec(Projectable(table_getindex_pr), table, ind)
table_getindex_spec(table, ind) = create_spec(Preprocess(table_getindex), table, ind)
