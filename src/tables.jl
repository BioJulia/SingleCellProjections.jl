get_id_colname_spec(table) = getindex_spec(get_colnames(table), 1)
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
