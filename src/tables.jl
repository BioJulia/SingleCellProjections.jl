# NB: Column names here are fixed and expected to be strings.
create_table_impl(args::Pair{String,<:Any}...) = DataFrame(args...; copycols=false)
create_table_impl_spec(args...) = create_spec(create_table_impl, args...; __version=v"0.1.0")

create_table_pr(action::Action, args::Pair{String,<:Any}...) = create_table_impl_spec(action(args)...)
create_table_spec(args...) = create_spec(Projectable(create_table_pr), args...)
Jobs.create_table(args...) = Job(create_table_spec(args...))


function table_to_compound_result(table)
	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(table))])
end


function table_from_compound_result(compound_result, colnames)
	cols = (name=>cached(compound_result, name) for name in colnames)
	create_table_impl_spec(cols...)
end
function table_from_compound_result(compound_result)
	colnames = fetched(cached(compound_result; return_keys=true))
	create_spec(Preprocess(table_from_compound_result), compound_result, colnames)
end



# TODO: Can we get rid of these? By good separation of Projectables vs "impl" Specs, these shouldn't be needed.
# TODO: Find better names for these?
is_impl(x) = !(x.f isa AbstractPreprocess)
forwarded_to_impl(spec) = forwarded(is_impl, spec)





# TODO: We have a common pattern for many utility functions taking tables as input.
#       Can we avoid code duplication?

get_colnames_fallback(table) = names(table)
function get_colnames_impl(table)
	if table isa Spec && table.f == create_table_impl
		return first.(table.args)
	else
		return create_spec(get_colnames_fallback, table; __version=v"0.1.0")
	end
end
get_colnames_impl_spec(table) =
	create_spec(Preprocess(get_colnames_impl), forwarded_to_impl(table))
get_colnames_pr(action, table) = get_colnames_impl_spec(action(table))
function get_colnames(table)
	if table isa Spec && table.f isa Projectable
		if table.f.f == create_table_pr
			return first.(table.args)
		else
			return create_spec(Projectable(get_colnames_pr), table)
		end
	end
	# Not a Projectable
	get_colnames_impl(table)
end
get_colnames_spec(table) = create_spec(Preprocess(get_colnames), table)
Jobs.get_colnames(table) = Job(get_colnames_spec(table))



get_id_colname_fallback(table) = only(names(table,1))
function get_id_colname_impl(table)
	if table isa Spec && table.f == create_table_impl
		return first(first(table.args)) # key of first column
	else
		return create_spec(get_id_colname_fallback, table; __version=v"0.1.0")
	end
end
get_id_colname_impl_spec(table) =
	create_spec(Preprocess(get_id_colname_impl), forwarded_to_impl(table))
get_id_colname_pr(action, table) = get_id_colname_impl_spec(action(table))
function get_id_colname(table)
	if table isa Spec && table.f isa Projectable
		if table.f.f == create_table_pr
			return first(first(table.args)) # key of first column
		else
			return create_spec(Projectable(get_id_colname_pr), table)
		end
	end
	# Not a Projectable
	get_id_colname_impl(table)
end
get_id_colname_spec(table) = create_spec(Preprocess(get_id_colname), table)
Jobs.get_id_colname(table) = Job(get_id_colname_spec(table))



function get_value_colname_fallback(table)
	len = ncol(table)
	len == 2 || throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
	only(names(table,2))
end
function get_value_colname_impl(table)
	if table isa Spec && table.f == create_table_impl
		len = length(table.args)
		len == 2 || throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
		return first(table.args[2]) # key of second column
	else
		return create_spec(get_value_colname_fallback, table; __version=v"0.1.0")
	end
end
get_value_colname_impl_spec(table) =
	create_spec(Preprocess(get_value_colname_impl), forwarded_to_impl(table))
get_value_colname_pr(action, table) = get_value_colname_impl_spec(action(table))
function get_value_colname(table)
	if table isa Spec && table.f isa Projectable
		if table.f.f == create_table_pr
			len = length(table.args)
			len == 2 || throw(ArgumentError("Expected annotation to have exactly two columns, but found $len columns."))
			return first(table.args[2]) # key of second column
		else
			return create_spec(Projectable(get_value_colname_pr), table)
		end
	end
	# Not a Projectable
	get_value_colname_impl(table)
end
get_value_colname_spec(table) = create_spec(Preprocess(get_value_colname), table)
Jobs.get_value_colname(table) = Job(get_value_colname_spec(table))





get_columns_fallback(table, colnames::String...) = select(table, [colnames...]; copycols=false)
function get_columns_impl(table, colnames::String...)
	if table isa Spec && table.f == create_table_impl
		table_colnames = first.(table.args)
		ind = indexin(colnames, table_colnames)
		any(isnothing, ind) && throw(ArgumentError("The following column names where not found: $(setdiff(colnames, table_colnames))"))
		return create_table_impl_spec(table.args[ind]...)
	else
		return create_spec(get_columns_fallback, table, colnames...; __version=v"0.1.0")
	end
end
get_columns_impl_spec(table, colnames::String...) =
	create_spec(Preprocess(get_columns_impl), forwarded_to_impl(table), colnames...)
get_columns_pr(action, table, colnames::String...) = get_columns_impl_spec(action(table), action(colnames)...)
function get_columns(table, colnames::String...)
	if table isa Spec && table.f isa Projectable
		if table.f.f == create_table_pr
			table_colnames = first.(table.args)
			ind = indexin(colnames, table_colnames)
			any(isnothing, ind) && throw(ArgumentError("The following column names where not found: $(setdiff(colnames, table_colnames))"))
			return create_table_spec(table.args[ind]...)
		else
			return create_spec(Projectable(get_columns_pr), table, colnames...)
		end
	end
	# Not a Projectable
	get_columns_impl(table, colnames...)
end
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

column_data_fallback(table, col) = table[!,col]
function column_data_impl(table, col)
	if table isa Spec && table.f == create_table_impl
		i = _col_ind(table, col)
		return table.args[i][2]
	else
		return create_spec(column_data_fallback, table, col; __version=v"0.1.0")
	end
end
column_data_impl_spec(table, col) =
	create_spec(Preprocess(column_data_impl), forwarded_to_impl(table), col)
column_data_pr(action, table, col) = column_data_impl_spec(action(table), col)
function column_data(table, col)
	if table isa Spec && table.f isa Projectable
		if table.f.f == create_table_pr
			i = _col_ind(table, col)
			return table.args[i][2]
		else
			return create_spec(Projectable(column_data_pr), table, col)
		end
	end
	# Not a Projectable
	column_data_impl(table, col)
end
column_data_spec(table, col) = create_spec(Preprocess(column_data), table, col)
Jobs.column_data(table, col) = Job(column_data_spec(table, col))


table_nrow(table) = length_spec(column_data_spec(table,1))
table_nrow_spec(table) = create_spec(Preprocess(table_nrow), table)
Jobs.table_nrow(table) = Job(table_nrow_spec(table))



_add_column_length_error(n1, n2, name) = throw(ArgumentError("Expected column \"$name\" to have length $n1, but got $n2."))
_add_column_length_error_spec(n1, n2, name) = create_spec(_add_column_length_error, n1, n2, name)


add_column_fallback(table, name, column) = insertcols(table, name=>column; copycols=false)
function add_column_impl(table, name, column)
	if table isa Spec && table.f == create_table_impl
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
		return create_spec(add_column_fallback, table, name, column; __version=v"0.1.0")
	end
end
add_column_impl_spec(table, name, column) =
	create_spec(Preprocess(add_column_impl), forwarded_to_impl(table), name, column)
add_column_pr(action, table, name, column) = add_column_impl_spec(action(table), name, action(column))
function add_column(table, name, column)
	if table isa Spec && table.f isa Projectable
		if table.f.f == create_table_pr
			# Check that there is no column with that name
			if name in (k for (k,_) in table.args)
				throw(ArgumentError("A column with the name \"$name\" already exists."))
			end

			result = create_table_spec(table.args..., name=>column)

			# Check that the length of the new column matches the old
			n1 = table_nrow_spec(table)
			n2 = length_spec(column)
			cond = isequal_spec(n1, n2)
			return ifelse_spec(cond, result, _add_column_length_error_spec(n1,n2,name))
		else
			return create_spec(Projectable(add_column_pr), table, name, column)
		end
	end
	# Not a Projectable
	add_column_impl(table, name, column)
end
add_column_spec(table, name, column) = create_spec(Preprocess(add_column), table, name, column)
Jobs.add_column(table, name, column) = Job(add_column_spec(table, name, column))






table_getindex_fallback(table, ind) = table[ind,:]
function table_getindex_impl(table, ind)
	ind === Colon() && return table # Indexing is a no-op, just return the table

	if table isa Spec && table.f == create_table_impl
		cols = (k=>getindex_impl_spec(v, ind) for (k,v) in table.args)
		return create_table_impl_spec(cols...)
	else
		return create_spec(table_getindex_fallback, table, prefetched(ind); __version=v"0.1.0")
	end
end
table_getindex_impl_spec(table, ind) =
	create_spec(Preprocess(table_getindex_impl), forwarded_to_impl(table), fetched(ind))
function table_getindex_pr(action, table, ind)
	table_p = action(table)
	result = table_getindex_impl_spec(table_p, action(ind))

	if action isa Projection && !(ind isa Spec)
		cond = isequal_spec(table, table_p)
		result = ifelse_spec(cond, result, _getindex_error_spec(ind))
	end

	result
end
function table_getindex(table, ind)
	if table isa Spec && table.f isa Projectable
		if table.f.f == create_table_pr
			cols = (k=>getindex_spec(v, ind) for (k,v) in table.args)
			return create_table_spec(cols...)
		else
			return create_spec(Projectable(table_getindex_pr), table, ind)
		end
	end
	# Not a Projectable
	table_getindex_impl(table, ind)
end
table_getindex_spec(table, ind) = create_spec(Preprocess(table_getindex), table, ind)



# TODO: Try to implementl table_leftjoin in terms of other, more basic functions that we just preprocess to
function table_leftjoin_fallback(a::DataFrame, b::DataFrame)
	ind = indexin_impl(a[!,1], b[!,1]; not_found=:nothing)
	b = select(b, Not(1); copycols=false) # get rid of ID column
	b_reordered = mapcols(b) do col
		getindex_or_missing(col, ind)
	end
	hcat(a, b_reordered; copycols=false)
end
function table_leftjoin_impl(a, b)
	if a isa Spec && a.f == create_table_impl && b isa Spec && b.f == create_table_impl
		idcol_a, ids_a = a.args[1]
		idcol_b, ids_b = b.args[1]
		idcol_a != idcol_b && throw(ArgumentError("ID column names \"$idcol_a\" and \"$idcol_b\" do not match."))

		names_a = first.(a.args)
		names_b = first.(b.args)
		common_names = intersect(@view(names_a[2:end]), @view(names_b[2:end]))
		isempty(common_names) || throw(ArgumentError("Table columns must be different (except ID column), found these common columns: $common_names"))

		ind_spec = indexin_impl_spec(ids_a, ids_b; not_found=:nothing)
		b_cols = ReproducibleJobs.unsafe_unmanage(b.args)
		joined_cols = [k=>getindex_or_missing_impl_spec(v,ind_spec) for (k,v) in @view(b_cols[2:end])]
		create_table_impl_spec(a.args..., joined_cols...)
	else
		return create_spec(table_leftjoin_fallback, a, b; __version=v"0.1.0")
	end
end
table_leftjoin_impl_spec(a, b) =
	create_spec(Preprocess(table_leftjoin_impl), forwarded_to_impl(a), forwarded_to_impl(b))
table_leftjoin_pr(action, a, b) = table_leftjoin_impl_spec(action(a), action(b))
function table_leftjoin(a, b)
	# Both Projectable(create_table_pr)
	# -> create_table_spec
	# At least one Projectable
	# -> Projectable
	# otherwise
	# -> impl

	if a isa Spec && a.f === Projectable(create_table_pr) && b isa Spec && b.f === Projectable(create_table_pr)
		idcol_a, ids_a = a.args[1]
		idcol_b, ids_b = b.args[1]
		idcol_a != idcol_b && throw(ArgumentError("ID column names \"$idcol_a\" and \"$idcol_b\" do not match."))

		names_a = first.(a.args)
		names_b = first.(b.args)
		common_names = intersect(@view(names_a[2:end]), @view(names_b[2:end]))
		isempty(common_names) || throw(ArgumentError("Table columns must be different (except ID column), found these common columns: $common_names"))

		ind_spec = indexin_spec(ids_a, ids_b; not_found=:nothing)
		b_cols = ReproducibleJobs.unsafe_unmanage(b.args)
		joined_cols = [k=>getindex_or_missing_spec(v,ind_spec) for (k,v) in @view(b_cols[2:end])]
		create_table_spec(a.args..., joined_cols...)
	elseif (a isa Spec && a.f isa Projectable) || (b isa Spec && b.f isa Projectable)
		return create_spec(Projectable(table_leftjoin_pr), a, b)
	else
		return table_leftjoin_impl(a, b)
	end
end
table_leftjoin_spec(a, b) = create_spec(Preprocess(table_leftjoin), a, b)
Jobs.table_leftjoin(a, b) = Job(table_leftjoin_spec(a, b))






function intersect_ids_fallback(a, b)
	ncol(a) != 1 && throw(ArgumentError("Expected `a` to have exactly one column."))
	ncol(b) != 1 && throw(ArgumentError("Expected `b` to have exactly one column."))
	a_name = only(names(a,1))
	b_name = only(names(b,1))
	a_name != b_name && throw(ArgumentError("ID column names \"$a_name\" and \"$b_name\" do not match."))
	DataFrame(a_name => intersect(a[!,1], b[!,1]); copycols=false)
end
function intersect_ids_impl(a, b)
	if a isa Spec && a.f == create_table_impl && b isa Spec && b.f == create_table_impl
		length(a.args) != 1 && throw(ArgumentError("Expected `a` to have exactly one column."))
		length(b.args) != 1 && throw(ArgumentError("Expected `b` to have exactly one column."))

		a_name,a_values = a.args[1]
		b_name,b_values = b.args[1]
		a_name != b_name && throw(ArgumentError("ID column names \"$a_name\" and \"$b_name\" do not match."))

		values = intersect_impl_spec(a_values, b_values)
		return create_table_impl_spec(a_name=>values)
	else
		return create_spec(intersect_ids_fallback, a, b; __version=v"0.1.0")
	end
end
intersect_ids_impl_spec(a, b) =
	create_spec(Preprocess(intersect_ids_impl), forwarded_to_impl(a), forwarded_to_impl(b))
intersect_ids_pr(action, a, b) = intersect_ids_impl_spec(action(a), action(b))
function intersect_ids(a, b)
	# Both Projectable(create_table_pr)
	# -> create_table_spec
	# At least one Projectable
	# -> Projectable
	# otherwise
	# -> impl

	if a isa Spec && a.f === Projectable(create_table_pr) && b isa Spec && b.f === Projectable(create_table_pr)
		length(a.args) != 1 && throw(ArgumentError("Expected `a` to have exactly one column."))
		length(b.args) != 1 && throw(ArgumentError("Expected `b` to have exactly one column."))

		a_name,a_values = a.args[1]
		b_name,b_values = b.args[1]
		a_name != b_name && throw(ArgumentError("ID column names \"$a_name\" and \"$b_name\" do not match."))

		values = intersect_spec(a_values, b_values)
		return create_table_spec(a_name=>values)
	elseif (a isa Spec && a.f isa Projectable) || (b isa Spec && b.f isa Projectable)
		return create_spec(Projectable(intersect_ids_pr), a, b)
	else
		return intersect_ids_impl(a, b)
	end
end
intersect_ids_spec(a, b) = create_spec(Preprocess(intersect_ids), a, b)

