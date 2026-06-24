# NB: Column names here are fixed and expected to be strings.
create_table(args::Pair{String,<:Any}...) = DataFrame(args...; copycols=false)
create_table_job(args...) = create_job(create_table, args...; __version=v"0.1.0")
"""
    Jobs.create_table(col1 => values1, col2 => values2, ...) -> Job

Create a new table `Job` from column name/value pairs.
"""
Jobs.create_table(args...) = create_table_job(args...)

is_create_table(x::SpecRef) = x.f == create_table
is_create_table(::Any) = false


table_to_compound_result(table) = CompoundResult(; pairs(eachcol(table))...)


# With known colnames
function table_from_compound_result(compound_result, colnames)
	cols = (name=>cached(compound_result, name) for name in colnames)
	create_table_job(cols...)
end

table_from_compound_result_pre(::Preprocessing, compound_result, colnames) =
	table_from_compound_result(compound_result, colnames)
function table_from_compound_result_pr(action::Action, compound_result)
	compound_result = action(compound_result)
	colnames = fetched(cached(compound_result; return_keys=true))
	create_job(Preprocess(table_from_compound_result_pre), compound_result, colnames) # we must preprocess so that colnames are fetched
end

# To handle when colnames differ due to projection
table_from_compound_result(compound_result) =
	create_job(Projectable(table_from_compound_result_pr), compound_result)




_get_ncol(table::DataFrame) = ncol(table)
_get_ncol(table::SpecRef) = length(table.args) # NB: only valid for create_table spec

function _check_ncol(table; require_n_cols=nothing)
	if require_n_cols !== nothing
		len = _get_ncol(table)
		len != require_n_cols && throw(ArgumentError("Expected table to have exactly $require_n_cols columns, but found $len columns."))
	end
	nothing
end

_get_colnames_fallback(table) = names(table)
_get_colnames_fallback(table, ind::Int) = only(names(table,ind))

function get_colnames_fallback(table, args...; kwargs...)
	_check_ncol(table; kwargs...)
	_get_colnames_fallback(table, args...)
end

_get_colnames(table) = collect(first.(table.args))
_get_colnames(table, ind) = first(table.args[ind])


function get_colnames(::Preprocessing{E}, table, args...; kwargs...) where E
	if is_create_table(table)
		_check_ncol(table; kwargs...)
		_get_colnames(table, args...)
	elseif E
		create_job(Preprocess{false}(get_colnames), table, args...; kwargs...)
	else
		create_job(get_colnames_fallback, table, args...; kwargs..., __version=v"0.1.0")
	end
end

get_colnames_job(table; kwargs...) = create_job(Preprocess(get_colnames), table; kwargs...)
get_colnames_job(table, ind::Int; kwargs...) = create_job(Preprocess(get_colnames), table, ind; kwargs...)
"""
    Jobs.get_colnames(table; kwargs...) -> Job

Return the column names of `table`.

See also [`Jobs.get_id_colname`](@ref), [`Jobs.get_value_colname`](@ref).
"""
Jobs.get_colnames(table, args...; kwargs...) = get_colnames_job(table, args...; kwargs...)




# Should add another layer of Preprocessing so that we see `get_id_colname` when forwarding Specs one step at a time?
get_id_colname_job(table) = create_job(Preprocess(get_colnames), table, 1)
"""
    Jobs.get_id_colname(table) -> Job

Return the name of the first (ID) column of `table`.

See also [`Jobs.get_colnames`](@ref), [`Jobs.get_value_colname`](@ref).
"""
Jobs.get_id_colname(table) = get_id_colname_job(table)

# Should add another layer of Preprocessing so that we see `get_value_colname` when forwarding Specs one step at a time?
get_value_colname_job(table) = create_job(Preprocess(get_colnames), table, 2; require_n_cols=2)
"""
    Jobs.get_value_colname(table) -> Job

Return the name of the second (value) column of `table`. Requires the table to have
exactly two columns.

See also [`Jobs.get_colnames`](@ref), [`Jobs.get_id_colname`](@ref).
"""
Jobs.get_value_colname(table) = get_value_colname_job(table)




function _colnames_to_colind(table, colnames::String...)
	table_colnames = first.(table.args)
	# ind = indexin(colnames, table_colnames)
	ind = indexin(colnames, collect(table_colnames)) # Refactoring TODO: avoid converting tuple to arrays?
	any(isnothing, ind) && throw(ArgumentError("The following column names where not found: $(setdiff(colnames, table_colnames))"))
	convert(Vector{Int}, ind)
end
_colnames_to_colind(table, colind::Int...) = collect(colind)

function get_columns_fallback(table, colnames...; kwargs...)
	_check_ncol(table; kwargs...)
	select(table, [colnames...]; copycols=false)
end
function get_columns(::Preprocessing{E}, table, colnames...; kwargs...) where E
	if is_create_table(table)
		_check_ncol(table; kwargs...)
		ind = _colnames_to_colind(table, colnames...)
		create_table_job(table.args[ind]...)
	elseif E
		create_job(Preprocess{false}(get_columns), table, colnames...; kwargs...)
	else
		create_job(get_columns_fallback, table, colnames...; kwargs..., __version=v"0.1.0")
	end
end
get_columns_job(table, colname1, colnames...; kwargs...) = create_job(Preprocess(get_columns), table, colname1, colnames...; kwargs...)
"""
    Jobs.get_columns(table, colnames...) -> Job

Select specific columns from `table` by name or index.

See also [`Jobs.id_column`](@ref), [`Jobs.value_column`](@ref).
"""
Jobs.get_columns(table, colname1, colnames...; kwargs...) = get_columns_job(table, colname1, colnames...; kwargs...)



id_column(::Preprocessing, table) = get_columns_job(table, 1)
id_column_job(table) = create_job(Preprocess(id_column), table)
"""
    Jobs.id_column(table) -> Job

Extract the first (ID) column of `table` as a single-column table.

See also [`Jobs.value_column`](@ref), [`Jobs.id_column_data`](@ref).
"""
Jobs.id_column(table) = id_column_job(table)

value_column(::Preprocessing, table) = get_columns_job(table, 2; require_n_cols=2)
value_column_job(table) = create_job(Preprocess(value_column), table)
"""
    Jobs.value_column(table) -> Job

Extract the second (value) column of `table` as a single-column table.

See also [`Jobs.id_column`](@ref), [`Jobs.value_column_data`](@ref).
"""
Jobs.value_column(table) = value_column_job(table)

annotation(::Preprocessing, table, colname) = get_columns_job(table, fetched(get_id_colname_job(table)), colname) # If we add support for mixed column indexing, this could be (1, colname)
annotation_job(table, colname) = create_job(Preprocess(annotation), table, colname)
"""
    Jobs.annotation(table, colname) -> Job

Extract the ID column and the column named `colname` from `table`, returning a two-column
table. Useful for passing annotations to filtering or covariate specification.
"""
Jobs.annotation(table, colname) = annotation_job(table, colname)



_col_ind(::Any, col::Integer) = col
function _col_ind(table, col::String)
	i = findfirst(p->isequal(p[1], col), table.args)
	i === nothing && throw(KeyError(col))
	i
end

function column_data_fallback(table, col::Union{String,Int}; kwargs...)
	_check_ncol(table; kwargs...)
	table[!,col]
end
function column_data(::Preprocessing{E}, table, col; kwargs...) where E
	if is_create_table(table)
		_check_ncol(table; kwargs...)
		i = _col_ind(table, col)
		table.args[i][2]
	elseif E
		create_job(Preprocess{false}(column_data), table, col; kwargs...)
	else
		create_job(column_data_fallback, table, col; kwargs..., __version=v"0.1.0")
	end
end
column_data_job(table, col; kwargs...) = create_job(Preprocess(column_data), table, col; kwargs...)
"""
    Jobs.column_data(table, col; kwargs...) -> Job

Return the values of column `col` from `table` as a vector.

See also [`Jobs.id_column_data`](@ref), [`Jobs.value_column_data`](@ref).
"""
Jobs.column_data(table, col; kwargs...) = column_data_job(table, col; kwargs...)




id_column_data(::Preprocessing, table) = column_data_job(table, 1)
id_column_data_job(table) = create_job(Preprocess(id_column_data), table)
"""
    Jobs.id_column_data(table) -> Job

Return the vector of IDs (first column) from `table`.

See also [`Jobs.column_data`](@ref), [`Jobs.value_column_data`](@ref).
"""
Jobs.id_column_data(table) = id_column_data_job(table)

value_column_data(::Preprocessing, table) = column_data_job(table, 2; require_n_cols=2)
value_column_data_job(table) = create_job(Preprocess(value_column_data), table)
"""
    Jobs.value_column_data(table) -> Job

Return the values (second column) from `table` as a vector. Requires the table to have
exactly two columns.

See also [`Jobs.column_data`](@ref), [`Jobs.id_column_data`](@ref).
"""
Jobs.value_column_data(table) = value_column_data_job(table)





table_nrow(::Preprocessing, table) = length_job(column_data_job(table,1))
table_nrow_job(table) = create_job(Preprocess(table_nrow), table)
"""
    Jobs.table_nrow(table) -> Job

Return the number of rows in `table`.

See also [`Jobs.table_ncol`](@ref).
"""
Jobs.table_nrow(table) = table_nrow_job(table)



table_ncol_fallback(table) = ncol(table)
function table_ncol(::Preprocessing{E}, table) where E
	if is_create_table(table)
		length(table.args)
	elseif E
		create_job(Preprocess{false}(table_ncol), table)
	else
		create_job(table_ncol_fallback, table; __version=v"0.1.0")
	end
end
table_ncol_job(table) = create_job(Preprocess(table_ncol), table)
"""
    Jobs.table_ncol(table) -> Job

Return the number of columns in `table`.

See also [`Jobs.table_nrow`](@ref).
"""
Jobs.table_ncol(table) = table_ncol_job(table)



_add_column_length_error(n1, n2, name) = throw(ArgumentError("Expected column \"$name\" to have length $n1, but got $n2."))
_add_column_length_error_job(n1, n2, name) = create_job(_add_column_length_error, n1, n2, name)

function _add_column_validated(table, name, column)
	# Check that there is no column with that name
	if name in (k for (k,_) in table.args)
		throw(ArgumentError("A column with the name \"$name\" already exists."))
	end

	result = create_table_job(table.args..., name=>column)

	# Check that the length of the new column matches the old
	n1 = table_nrow_job(table)
	n2 = length_job(column)
	cond = isequal_job(n1, n2)
	ifelse_pr_job(cond, result, _add_column_length_error_job(n1,n2,name))
end

add_column_fallback(table, name, column) = insertcols(table, name=>column; copycols=false)
function add_column(::Preprocessing{E}, table, name, column) where E
	if is_create_table(table)
		_add_column_validated(table, name, column)
	elseif E
		create_job(Preprocess{false}(add_column), table, name, column)
	else
		create_job(add_column_fallback, table, name, column; __version=v"0.1.0")
	end
end
add_column_job(table, name, column) = create_job(Preprocess(add_column), table, name, column)
"""
    Jobs.add_column(table, name, column) -> Job

Add a column named `name` with values `column` to `table`.
The length of `column` must match the number of rows in `table`.

See also [`Jobs.table_hcat`](@ref), [`Jobs.add_var_column`](@ref), [`Jobs.add_obs_column`](@ref).
"""
Jobs.add_column(table, name, column) = add_column_job(table, name, column)



_table_hcat_nrow_error(n) = throw(ArgumentError("Expected number of tables rows to match, but got $(join(n, ", ", " and "))."))
_table_hcat_nrow_error_job(n) = create_job(_table_hcat_nrow_error, n)

function _table_hcat_validated(args...)
	names = vcat((first.(a.args) for a in args)...)
	common_names = [name for (name,count) in StatsBase.countmap(names) if count>1]
	isempty(common_names) || throw(ArgumentError("Table column names must be different, found these common names: $common_names"))

	result = create_table_job(Iterators.flatten(getproperty.(args,:args))...)

	# Check that the number of rows in all tables match
	n = table_nrow_job.(args)
	cond = allequal_job(n)
	ifelse_pr_job(cond, result, _table_hcat_nrow_error_job(n))
end


table_hcat_fallback(args::DataFrame...) = hcat(args...; copycols=false)
function table_hcat(::Preprocessing{E}, args...) where E
	if all(is_create_table, args)
		_table_hcat_validated(args...)
	elseif E
		create_job(Preprocess{false}(table_hcat), args...) # try again with late preprocessing (i.e. after projectables has been hanlded)
	else
		create_job(table_hcat_fallback, args...; __version=v"0.1.0")
	end
end

# TODO: Refactor to take a vector instead? Better for the compiler if there are many arguments.
table_hcat_job(a, args...) = create_job(Preprocess(table_hcat), a, args...)
"""
    Jobs.table_hcat(a, tables...) -> Job

Horizontally concatenate tables. All tables must have the same number of rows and
matching row order.

See also [`Jobs.table_leftjoin`](@ref), [`Jobs.add_column`](@ref).
"""
Jobs.table_hcat(a, args...) = table_hcat_job(a, args...)





# Another try
table_getindex_fallback(table, ind) = table[ind,:]

function table_getindex_pr(action, table, ind)
	table_p = action(table)
	result = create_job(Preprocess{false}(table_getindex), table_p, action(ind))

	if action isa Projection && !(ind isa SpecRef) # TODO: Fix, this will trigger even if ind is replaced by the action, which it shouldn't - maybe hard to avoid?
		cond = isequal_job(table, table_p)
		result = ifelse_job(cond, result, _getindex_error_job(ind))
	end

	result
end
table_getindex_pr_job(table, ind) = create_job(Projectable(table_getindex_pr), table, ind)

function table_getindex(::Preprocessing{E}, table, ind) where E
	if !E && ind == Colon()
		table # Projections have been handled, so indexing by `:` is OK
	elseif is_create_table(table) # Move the operation to the columns if we can
		cols = (k=>getindex_job(v, ind) for (k,v) in table.args)
		create_table_job(cols...)
	elseif E # early is before projection, so we need to handle the projection
		table_getindex_pr_job(table, ind)
	else
		create_job(table_getindex_fallback, table, ind; __version=v"0.1.0")
	end


	# if is_create_table(table) # Move the operation to the columns if we can
	# 	cols = (k=>getindex_job(v, ind) for (k,v) in table.args)
	# 	create_table_job(cols...)
	# elseif E # early is before projection, so we need to handle the projection
	# 	table_getindex_pr_job(table, ind)
	# elseif ind == Colon() # Projections have been handled, so indexing by `:` will not be transformed to something else
	# 	table
	# else
	# 	create_job(table_getindex_fallback, table, ind; __version=v"0.1.0")
	# end
end
table_getindex_job(table, ind) = create_job(Preprocess(table_getindex), table, ind)





function _table_leftjoin(a, b)
	idcol_a, ids_a = a.args[1]
	idcol_b, ids_b = b.args[1]
	idcol_a != idcol_b && throw(ArgumentError("ID column names \"$idcol_a\" and \"$idcol_b\" do not match."))

	names_a = first.(a.args)
	names_b = first.(b.args)
	common_names = intersect(names_a[2:end], names_b[2:end])
	isempty(common_names) || throw(ArgumentError("Table columns must be different (except ID column), found these common columns: $common_names"))

	ind_job = indexin_job(ids_a, ids_b; not_found=:nothing)
	joined_cols = [k=>getindex_or_missing_job(v,ind_job) for (k,v) in b.args[2:end]]
	create_table_job(a.args..., joined_cols...)
end

function table_leftjoin_fallback(a::DataFrame, b::DataFrame)
	ind = indexin_impl(a[!,1], b[!,1]; not_found=:nothing)
	b = select(b, Not(1); copycols=false) # get rid of ID column
	b_reordered = mapcols(b) do col
		getindex_or_missing(col, ind)
	end
	hcat(a, b_reordered; copycols=false)
end

function table_leftjoin(::Preprocessing{E}, a, b) where E
	if is_create_table(a) && is_create_table(b)
		_table_leftjoin(a, b)
	elseif E
		create_job(Preprocess{false}(table_leftjoin), a, b) # try again with late preprocessing (i.e. after projectables has been hanlded)
	else
		create_job(table_leftjoin_fallback, a, b; __version=v"0.1.0")
	end
end

table_leftjoin_job(a, b) = create_job(Preprocess(table_leftjoin), a, b)
"""
    Jobs.table_leftjoin(a, b) -> Job

Left-join table `b` onto table `a` by their ID columns.

See also [`Jobs.table_hcat`](@ref), [`Jobs.annotate_var`](@ref), [`Jobs.annotate_obs`](@ref).
"""
Jobs.table_leftjoin(a, b) = table_leftjoin_job(a, b)




combine_column_values_fallback(table::DataFrame; kwargs...) = combine_vectors_impl(eachcol(table); kwargs...)
function combine_column_values(::Preprocessing{E}, table; kwargs...) where E
	if is_create_table(table)
		values = (v for (_,v) in table.args)
		combine_vectors_job(values...; kwargs...)
	elseif E
		create_job(Preprocess{false}(combine_column_values), table; kwargs...)
	else
		create_job(combine_column_values_fallback, table; kwargs..., __version=v"0.1.0")
	end
end

combine_column_values_job(table; kwargs...) = create_job(Preprocess(combine_column_values), table; kwargs...)



repeat_columns_fallback(table::DataFrame; kwargs...) = mapcols(v->repeat(v; kwargs...), table)
function repeat_columns(::Preprocessing{E}, table; kwargs...) where E
	if is_create_table(table)
		cols = (k=>repeat_job(v; kwargs...) for (k,v) in table.args)
		create_table_job(cols...)
	elseif E
		create_job(Preprocess{false}(repeat_columns), table; kwargs...)
	else
		create_job(repeat_columns_fallback, table; kwargs..., __version=v"0.1.0")
	end
end

repeat_columns_job(table; kwargs...) = create_job(Preprocess(repeat_columns), table; kwargs...)





function _intersect_ids(a, b)
	length(a.args) != 1 && throw(ArgumentError("Expected `a` to have exactly one column."))
	length(b.args) != 1 && throw(ArgumentError("Expected `b` to have exactly one column."))

	a_name,a_values = a.args[1]
	b_name,b_values = b.args[1]
	a_name != b_name && throw(ArgumentError("ID column names \"$a_name\" and \"$b_name\" do not match."))

	values = intersect_job(a_values, b_values)
	create_table_job(a_name=>values)
end

function intersect_ids_fallback(a, b)
	ncol(a) != 1 && throw(ArgumentError("Expected `a` to have exactly one column."))
	ncol(b) != 1 && throw(ArgumentError("Expected `b` to have exactly one column."))
	a_name = only(names(a,1))
	b_name = only(names(b,1))
	a_name != b_name && throw(ArgumentError("ID column names \"$a_name\" and \"$b_name\" do not match."))
	DataFrame(a_name => intersect(a[!,1], b[!,1]); copycols=false)
end
function intersect_ids(::Preprocessing{E}, a, b) where E
	if is_create_table(a) && is_create_table(b)
		_intersect_ids(a, b)
	elseif E
		create_job(Preprocess{false}(intersect_ids), a, b)
	else
		create_job(intersect_ids_fallback, a, b; __version=v"0.1.0")
	end
end
intersect_ids_job(a, b) = create_job(Preprocess(intersect_ids), a, b)




function transform_annotation_fallback(f, table; new_name=nothing)
	_check_ncol(table; require_n_cols=2)
	name = @something new_name only(names(table,2))
	DataFrame(only(names(table,1))=>table[!,1], name=>f.(table[!,2]); copycols=false)
end
function transform_annotation(::Preprocessing{E}, f, table; kwargs...) where E
	if is_create_table(table)
		_check_ncol(table; require_n_cols=2)
		name = @something get(kwargs, :new_name, nothing) table.args[2].first
		create_table_job(table.args[1], name => apply_broadcasted_job(f, table.args[2].second))
	elseif E
		create_job(Preprocess{false}(transform_annotation), f, table; kwargs...)
	else
		create_job(transform_annotation_fallback, f, table; kwargs..., __version=v"0.1.0")
	end
end
transform_annotation_job(f, table; kwargs...) = create_job(Preprocess(transform_annotation), f, table; kwargs...)
"""
    Jobs.transform_annotation(f, table; kwargs...) -> Job

Apply function `f` element-wise to the value column of `table`, returning a new table
with transformed values. The table must have exactly two columns (ID and value).
Use `new_name` to rename the value column.

(TODO: Example.)
"""
Jobs.transform_annotation(f, table; kwargs...) = transform_annotation_job(f, table; kwargs...)
