# TODO: Move some of these to a common utility specs
# args2vec_impl(::Type{T}, args...) where T = T[args...]
function args2vec_impl(::Type{T}, args...) where T
	T[args...]
end
args2vec_pr(action::Action, ::Type{T}, args...) where T =
	create_spec(args2vec_impl, T, action(args)...; __version=v"0.1.0")
args2vec_spec(::Type{T}, args...) where T =
	create_spec(Projectable(args2vec_pr), T, args...)


first_pr(action, v) = create_spec(first, action(v); __version=v"0.1.0")
first_spec(v) = create_spec(Projectable(first_pr), v)


issubset_pr(action, a, b) = create_spec(issubset, action(a), action(b); __version=v"0.1.0")
issubset_spec(a, b) = create_spec(Projectable(issubset_pr), a, b)

setdiff_pr(action, a, b) = create_spec(setdiff, action(a), action(b); __version=v"0.1.0")
setdiff_spec(a, b) = create_spec(Projectable(setdiff_pr), a, b)



get_id_colname_spec(table) = first_spec(get_colnames(table))
function Jobs.get_id_colname(table)
	Job(get_id_colname_spec(table))
end




_get_columns_error(colnames) =
	throw(ArgumentError("The following column names where not found: $colnames"))
_get_columns_error_spec(colnames) =
	create_spec(_get_columns_error, colnames; __version=v"0.1.0")


function get_columns(::ColNames, table, colnames...)
	table_colnames = get_colnames(table)
	result = args2vec_spec(String, colnames...)
	cond = issubset_spec(result, table_colnames)
	err = _get_columns_error_spec(setdiff_spec(result, table_colnames))
	ifelse_spec(cond, result, err)
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
