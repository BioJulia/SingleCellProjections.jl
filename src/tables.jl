get_id_colname_spec(table) = getindex_spec(get_colnames_spec(table), 1)
function Jobs.get_id_colname(table)
	Job(get_id_colname_spec(table))
end




_get_columns_error(colnames) =
	throw(ArgumentError("The following column names where not found: $colnames"))
_get_columns_error_pr(action, colnames) = create_spec(_get_columns_error, action(colnames); __version=v"0.1.0")
_get_columns_error_spec(colnames) =
	create_spec(Projectable(_get_columns_error_pr), colnames)


function get_columns(::ColNames, table, colnames...)
	table_colnames = get_colnames(table)
	result = args2vec_spec(String, colnames...)
	cond = issubset_spec(result, table_colnames)
	err = _get_columns_error_spec(setdiff_spec(result, table_colnames))
	ifelse_pr_spec(cond, result, err)
end
function get_columns(c::Col, table, colnames...)
	# Shoule we assert here too? (At the moment, we consider it enough to do it when retrieving the colnames.)
	get_col_spec(table, c.name)
end


get_columns_spec(table, colname1, colnames...) =
	create_spec(TableFunction(get_columns), table, colname1, colnames...)

function Jobs.get_columns(table, colname1, colnames...)
	Job(get_columns_spec(table, colname1, colnames...))
end



annotation(table, colname) = get_columns_spec(table, get_id_colname_spec(table), colname)
annotation_spec(table, colname) = create_spec(Preprocess(annotation), table, colname)
function Jobs.annotation(table, colname)
	Job(annotation_spec(table, colname))
end




_get_value_colname_error(len) =
	throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
_get_value_colname_error_pr(action, len) = create_spec(_get_value_colname_error, action(len); __version=v"0.1.0")
_get_value_colname_error_spec(len) =
	create_spec(Projectable(_get_value_colname_error_pr), len)


function get_value_colname_spec(table)
	# Ensure there are exactly two columns
	# Return the name of the second column
	colnames = get_colnames_spec(table)
	len = length_spec(colnames)
	cond = isequal_spec(len, 2)
	ifelse_pr_spec(cond, getindex_spec(colnames, 2), _get_value_colname_error_spec(len))
end
function Jobs.get_value_colname(table)
	Job(get_value_colname_spec(table))
end
