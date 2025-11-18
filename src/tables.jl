# NB: Column names here are fixed and expected to be strings.
create_table(args::Pair{String,<:Any}...) = DataFrame(args...; copycols=false)
create_table_spec(args...) = create_spec(create_table, args...; __version=v"0.1.0")
Jobs.create_table(args...) = Job(create_table_spec(args...))

is_create_table(x) = x isa Spec && x.f == create_table




function table_to_compound_result(table)
	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(table))])
end


function table_from_compound_result(compound_result, colnames)
	cols = (name=>cached(compound_result, name) for name in colnames)
	create_table_spec(cols...)
end
table_from_compound_result(::Preprocessing, compound_result, colnames) = table_from_compound_result(compound_result, colnames)
function table_from_compound_result(compound_result)
	colnames = fetched(cached(compound_result; return_keys=true))
	create_spec(Preprocess(table_from_compound_result), compound_result, colnames)
end




_get_ncol(table::DataFrame) = ncol(table)
_get_ncol(table::Spec) = length(table.args) # NB: only valid for create_table spec

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

_get_colnames(table) = first.(table.args)
_get_colnames(table, ind) = first(table.args[ind])


function get_colnames(::Preprocessing{E}, table, args...; kwargs...) where E
	if is_create_table(table)
		_check_ncol(table; kwargs...)
		_get_colnames(table, args...)
	elseif E
		create_spec(Preprocess{false}(get_colnames), table, args...; kwargs...)
	else
		create_spec(get_colnames_fallback, table, args...; kwargs..., __version=v"0.1.0")
	end
end

get_colnames_spec(table; kwargs...) = create_spec(Preprocess(get_colnames), table; kwargs...)
get_colnames_spec(table, ind::Int; kwargs...) = create_spec(Preprocess(get_colnames), table, ind; kwargs...)
Jobs.get_colnames(table, args...; kwargs...) = Job(get_colnames_spec(table, args...; kwargs...))




# Should add another layer of Preprocessing so that we see `get_id_colname` when forwarding Specs one step at a time?
get_id_colname_spec(table) = create_spec(Preprocess(get_colnames), table, 1)
Jobs.get_id_colname(table) = Job(get_id_colname_spec(table))

# Should add another layer of Preprocessing so that we see `get_value_colname` when forwarding Specs one step at a time?
get_value_colname_spec(table) = create_spec(Preprocess(get_colnames), table, 2; require_n_cols=2)
Jobs.get_value_colname(table) = Job(get_value_colname_spec(table))




function _colnames_to_colind(table, colnames::String...)
	table_colnames = first.(table.args)
	ind = indexin(colnames, table_colnames)
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
		create_table_spec(table.args[ind]...)
	elseif E
		create_spec(Preprocess{false}(get_columns), table, colnames...; kwargs...)
	else
		create_spec(get_columns_fallback, table, colnames...; kwargs..., __version=v"0.1.0")
	end
end
get_columns_spec(table, colname1, colnames...; kwargs...) = create_spec(Preprocess(get_columns), table, colname1, colnames...; kwargs...)
Jobs.get_columns(table, colname1, colnames...; kwargs...) = Job(get_columns_spec(table, colname1, colnames...; kwargs...))



id_column(::Preprocessing, table) = get_columns_spec(table, 1)
id_column_spec(table) = create_spec(Preprocess(id_column), table)
Jobs.id_column(table) = Job(id_column_spec(table))

value_column(::Preprocessing, table) = get_columns_spec(table, 2; require_n_cols=2)
value_column_spec(table) = create_spec(Preprocess(value_column), table)
Jobs.value_column(table) = Job(value_column_spec(table))

annotation(::Preprocessing, table, colname) = get_columns_spec(table, fetched(get_id_colname_spec(table)), colname) # If we add support for mixed column indexing, this could be (1, colname)
annotation_spec(table, colname) = create_spec(Preprocess(annotation), table, colname)
Jobs.annotation(table, colname) = Job(annotation_spec(table, colname))



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
		create_spec(Preprocess{false}(column_data), table, col; kwargs...)
	else
		create_spec(column_data_fallback, table, col; kwargs..., __version=v"0.1.0")
	end
end
column_data_spec(table, col; kwargs...) = create_spec(Preprocess(column_data), table, col; kwargs...)
Jobs.column_data(table, col; kwargs...) = Job(column_data_spec(table, col; kwargs...))




id_column_data(::Preprocessing, table) = column_data_spec(table, 1)
id_column_data_spec(table) = create_spec(Preprocess(id_column_data), table)
Jobs.id_column_data(table) = Job(id_column_data_spec(table))

value_column_data(::Preprocessing, table) = column_data_spec(table, 2; require_n_cols=2)
value_column_data_spec(table) = create_spec(Preprocess(value_column_data), table)
Jobs.value_column_data(table) = Job(value_column_data_spec(table))





table_nrow(::Preprocessing, table) = length_spec(column_data_spec(table,1))
table_nrow_spec(table) = create_spec(Preprocess(table_nrow), table)
Jobs.table_nrow(table) = Job(table_nrow_spec(table))



_add_column_length_error(n1, n2, name) = throw(ArgumentError("Expected column \"$name\" to have length $n1, but got $n2."))
_add_column_length_error_spec(n1, n2, name) = create_spec(_add_column_length_error, n1, n2, name)

function _add_column_validated(table, name, column)
	# Check that there is no column with that name
	if name in (k for (k,_) in table.args)
		throw(ArgumentError("A column with the name \"$name\" already exists."))
	end

	result = create_table_spec(table.args..., name=>column)

	# Check that the length of the new column matches the old
	n1 = table_nrow_spec(table)
	n2 = length_spec(column)
	cond = isequal_spec(n1, n2)
	ifelse_spec(cond, result, _add_column_length_error_spec(n1,n2,name))
end

add_column_fallback(table, name, column) = insertcols(table, name=>column; copycols=false)
function add_column(::Preprocessing{E}, table, name, column) where E
	if is_create_table(table)
		_add_column_validated(table, name, column)
	elseif E
		create_spec(Preprocess{false}(add_column), table, name, column)
	else
		create_spec(add_column_fallback, table, name, column; __version=v"0.1.0")
	end
end
add_column_spec(table, name, column) = create_spec(Preprocess(add_column), table, name, column)
Jobs.add_column(table, name, column) = Job(add_column_spec(table, name, column))



table_getindex_fallback(table, ind) = table[ind,:]
function table_getindex(::Preprocessing{E}, table, ind) where E
	if is_create_table(table)
		cols = (k=>getindex_spec(v, ind) for (k,v) in table.args)
		create_table_spec(cols...)
	elseif E
		# create_spec(Preprocess{false}(table_getindex), table, ind)
		create_spec(Preprocess{false}(table_getindex), table, fetched(ind))
	elseif ind == Colon() # Projections have been handled, so indexing by `:` will not be transformed to something else
		table
	else
		create_spec(table_getindex_fallback, table, ind; __version=v"0.1.0")
	end
end
table_getindex_spec(table, ind) = create_spec(Preprocess(table_getindex), table, ind)




function _table_leftjoin(a, b)
	idcol_a, ids_a = a.args[1]
	idcol_b, ids_b = b.args[1]
	idcol_a != idcol_b && throw(ArgumentError("ID column names \"$idcol_a\" and \"$idcol_b\" do not match."))

	names_a = first.(a.args)
	names_b = first.(b.args)
	common_names = intersect(@view(names_a[2:end]), @view(names_b[2:end]))
	isempty(common_names) || throw(ArgumentError("Table columns must be different (except ID column), found these common columns: $common_names"))

	ind_spec = indexin_spec2(ids_a, ids_b; not_found=:nothing)
	b_cols = ReproducibleJobs.unsafe_unmanage(b.args)
	joined_cols = [k=>getindex_or_missing_spec(v,ind_spec) for (k,v) in @view(b_cols[2:end])]
	create_table_spec(a.args..., joined_cols...)
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
		create_spec(Preprocess{false}(table_leftjoin), a, b) # try again with late preprocessing (i.e. after projectables has been hanlded)
	else
		create_spec(table_leftjoin_fallback, a, b; __version=v"0.1.0")
	end
end

table_leftjoin_spec(a, b) = create_spec(Preprocess(table_leftjoin), a, b)
Jobs.table_leftjoin(a, b) = Job(table_leftjoin_spec(a, b))




function _intersect_ids(a, b)
	length(a.args) != 1 && throw(ArgumentError("Expected `a` to have exactly one column."))
	length(b.args) != 1 && throw(ArgumentError("Expected `b` to have exactly one column."))

	a_name,a_values = a.args[1]
	b_name,b_values = b.args[1]
	a_name != b_name && throw(ArgumentError("ID column names \"$a_name\" and \"$b_name\" do not match."))

	values = intersect_spec2(a_values, b_values)
	create_table_spec(a_name=>values)
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
		create_spec(Preprocess{false}(intersect_ids), a, b)
	else
		create_spec(intersect_ids_fallback, a, b; __version=v"0.1.0")
	end
end
intersect_ids_spec(a, b) = create_spec(Preprocess(intersect_ids), a, b)
