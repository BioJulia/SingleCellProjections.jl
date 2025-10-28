# _to_string_pair((k,v)::Pair{K,V}) where {K,V} = convert(String,k)=>v

# NB: Column names here are fixed and expected to be strings.
create_table_impl(args::Pair{String,<:Any}...) = DataFrame(args...)
# create_table_impl_spec(args::Pair...) = create_spec(create_table_impl, _to_string_pair.(args)...; __version=v"0.0.1")
create_table_impl_spec(args...) = create_spec(create_table_impl, args...; __version=v"0.0.1")

create_table_pr(action::Action, args::Pair{String,<:Any}...) = create_table_impl_spec(action(args)...)
# create_table_spec(args::Pair...) = create_spec(Projectable(create_table_pr), _to_string_pair.(args)...)
create_table_spec(args...) = create_spec(Projectable(create_table_pr), args...)



# TODO: We see a common pattern in all these functions
#       And there will be other functions than create_table_* to handle.
#       So design some helper functions for e.g. extracting colnames from a "Table" spec.
#       And use those instead.
#       Handling both the case when the extracted is a Spec and when it is a value.



function get_colnames_pr(action, table)
	table = action(table)
	if table isa Spec && table.f in (create_table_pr, create_table_impl)
		return first.(table.args)
	else
		error("Not implemented")
	end
end

get_colnames(table_spec) = create_spec(Projectable(get_colnames_pr), table_spec)
get_colnames_spec(table) = create_spec(Preprocess(get_colnames), table)
Jobs.get_colnames(table) = Job(get_colnames_spec(table))



function get_id_colname_pr(action, table)
	table = action(table)
	if table isa Spec && table.f in (create_table_pr, create_table_impl)
		return first(first(table.args)) # key of first column
	else
		error("Not implemented")
	end
end

get_id_colname(table_spec) = create_spec(Projectable(get_id_colname_pr), table_spec)
get_id_colname_spec(table) = create_spec(Preprocess(get_id_colname), table)
Jobs.get_id_colname(table) = Job(get_id_colname_spec(table))



function get_value_colname_pr(action, table)
	table = action(table)
	if table isa Spec && table.f in (create_table_pr, create_table_impl)
		len = length(table.args)
		len == 2 || throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
		return first(table.args[2]) # key of second column
	else
		error("Not implemented")
	end
end

get_value_colname(table_spec) = create_spec(Projectable(get_value_colname_pr), table_spec)
get_value_colname_spec(table) = create_spec(Preprocess(get_value_colname), table)
Jobs.get_value_colname(table) = Job(get_value_colname_spec(table))



function get_columns_pr(action, table, colnames::String...)
	table = action(table)
	if table isa Spec && table.f in (create_table_pr, create_table_impl)
		table_colnames = first.(table.args)
		ind = indexin(colnames, table_colnames)
		if any(isnothing, ind)
			throw(ArgumentError("The following column names where not found: $(setdiff(colnames, table_colnames))"))
		end

		create_table_impl_spec(table.args[ind]...)
	else
		error("Not implemented")
	end
end

get_columns(table_spec, colnames...) = create_spec(Projectable(get_columns_pr), table_spec, colnames...)
get_columns_spec(table, colname1, colnames...) = create_spec(Preprocess(get_columns), table, colname1, colnames...)
Jobs.get_columns(table, colname1, colnames...) = Job(get_columns_spec(table, colname1, colnames...))



annotation(table, colname) = get_columns_spec(table, fetched(get_id_colname_spec(table)), colname)
annotation_spec(table, colname) = create_spec(Preprocess(annotation), table, colname)
Jobs.annotation(table, colname) = Job(annotation_spec(table, colname))





# --- Old ---

# get_id_colname_spec(table) = getindex_spec(get_colnames_spec(table), 1)
# function Jobs.get_id_colname(table)
# 	Job(get_id_colname_spec(table))
# end




# _get_columns_error(colnames) =
# 	throw(ArgumentError("The following column names where not found: $colnames"))
# _get_columns_error_pr(action, colnames) = create_spec(_get_columns_error, action(colnames); __version=v"0.1.0")
# _get_columns_error_spec(colnames) =
# 	create_spec(Projectable(_get_columns_error_pr), colnames)


# function get_columns(::ColNames, table, colnames...)
# 	table_colnames = get_colnames(table)
# 	result = args2vec_spec(String, colnames...)
# 	cond = issubset_spec(result, table_colnames)
# 	err = _get_columns_error_spec(setdiff_spec(result, table_colnames))
# 	ifelse_pr_spec(cond, result, err)
# end
# function get_columns(c::Col, table, colnames...)
# 	# Shoule we assert here too? (At the moment, we consider it enough to do it when retrieving the colnames.)
# 	get_col_spec(table, c.name)
# end


# get_columns_spec(table, colname1, colnames...) =
# 	create_spec(TableFunction(get_columns), table, colname1, colnames...)

# function Jobs.get_columns(table, colname1, colnames...)
# 	Job(get_columns_spec(table, colname1, colnames...))
# end



# annotation(table, colname) = get_columns_spec(table, get_id_colname_spec(table), colname)
# annotation_spec(table, colname) = create_spec(Preprocess(annotation), table, colname)
# function Jobs.annotation(table, colname)
# 	Job(annotation_spec(table, colname))
# end




# _get_value_colname_error(len) =
# 	throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
# _get_value_colname_error_pr(action, len) = create_spec(_get_value_colname_error, action(len); __version=v"0.1.0")
# _get_value_colname_error_spec(len) =
# 	create_spec(Projectable(_get_value_colname_error_pr), len)


# function get_value_colname_spec(table)
# 	# Ensure there are exactly two columns
# 	# Return the name of the second column
# 	colnames = get_colnames_spec(table)
# 	len = length_spec(colnames)
# 	cond = isequal_spec(len, 2)
# 	ifelse_pr_spec(cond, getindex_spec(colnames, 2), _get_value_colname_error_spec(len))
# end
# function Jobs.get_value_colname(table)
# 	Job(get_value_colname_spec(table))
# end
