abstract type TableField end # Find a better name?
struct ColNames <: TableField end
struct Col <: TableField
	name::String
end



create_table_impl(args::Pair...) = DataFrame(args...)
create_table_impl_spec(args::Pair...) = create_spec(create_table_impl, args...; __version=v"0.0.1")

create_table_pr(action::Action, args::Pair...) = create_table_impl_spec(action(args)...)
create_table_pr_spec(args::Pair...) = create_spec(Projectable(create_table_pr), args...)




# These are needed by get_colnames/get_col
setup_table(::ColNames, ::typeof(DataFrame), spec) =
	String[k for (k,v) in spec.args]
function setup_table(c::Col, ::typeof(DataFrame), spec)
	for (k,v) in spec.args
		k == c.name && return v
	end
	throw(KeyError(c.name))
end


# This changes get_field(project()) to project(get_field())
function setup_table(f::TableField, ::typeof(project), spec)
	onto = get_spec(f, spec.args[1])
	create_project_spec(onto, spec.args[2:end]...)
end



# WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
setup_table(f::TableField, t::TableFunction{F}, spec) where F = t.f(f, spec.args...; spec.kwargs...) # Should this be ColNames only?
setup_table(col::Col, t::ColNamesTableFunction{F}, spec) where F = t.f(col, spec.args...; spec.kwargs...)


function get_colnames_pr(action::Action, table_spec)
	@assert is_table_spec(table_spec) # TODO: We might want to relax this later
	action(setup_table(ColNames(), table_spec))
end
function get_col_pr(action::Action, table_spec, name)
	@assert is_table_spec(table_spec) # TODO: We might want to relax this later
	action(setup_table(Col(name), table_spec))
end


get_colnames(table_spec) = create_spec(Projectable(get_colnames_pr), table_spec)
get_col(table_spec, name) = create_spec(Projectable(get_col_pr), table_spec, name)

get_colnames_spec(x) = create_spec(Preprocess(get_colnames), x)
Jobs.get_colnames(x) = Job(get_colnames_spec(x))

get_col_spec(x, name) = create_spec(Preprocess(get_col), x, name)
Jobs.get_col(x, name) = Job(get_col_spec(x, name))

get_spec(::ColNames, x) = get_colnames_spec(x)
get_spec(c::Col, x) = get_col_spec(x, c.name)




is_table_spec(::Any) = false
function is_table_spec(sa::SpecArgs)
	f = sa.f

	# @assert !(f isa ColNamesTableFunction) # Testing


	f isa TableFunction && return true
	f isa ColNamesTableFunction && return true # Should this be here?
	f == create_table_impl && return true
	if f == project
		onto = sa.args[1]
		return is_table_spec(onto)
	end
	# TODO: Are there more cases that should return true?
	return false
end
is_table_spec(spec::Spec) = is_table_spec(spec.ro.value)



function _table_from_colnames(f::F, colnames, args...; kwargs...) where F
	cols = (name=>f(Col(name), args...; kwargs...) for name in colnames)
	# create_table_pr_spec(cols...) # TODO: This should need to be Projectable. Handled through project(::TableFunction) and project(::ColNamesTableFunction)
	create_table_impl_spec(cols...)
end



# for dispatch
setup_table(f::TableField, spec::Spec) = setup_table(f, spec.f, spec)



# This evaluates the TableFunction
function (d::TableFunction{F})(args...; kwargs...) where F
	colnames = d.f(ColNames(), args...; kwargs...) # can be a spec or just a list of names

	# We need to preprocess once to fetch the colnames.
	# This also means that colnames can be Projected.
	create_spec(ColNamesTableFunction(d.f), fetched(colnames), args...; kwargs...)

	# TODO: Should we do this? It's just a small shortcut for when colnames are not a Spec. But it should work fine with projections too.
	# if colnames isa Union{Spec,Job}
	# 	# We need to preprocess once to fetch the colnames
	# 	create_spec(ColNamesTableFunction(d.f), fetched(colnames), args...; kwargs...)
	# else
	# 	# colnames are not a spec, just setup directly
	# 	_table_from_colnames(d.f, colnames, args...; kwargs...)
	# end
end


function (d::ColNamesTableFunction{F})(colnames, args...; kwargs...) where F
	_table_from_colnames(d.f, colnames, args...; kwargs...)
end





function try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_colnames)}, k::Spec, v)
	if is_table_spec(k)
		# Replace the inner spec
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_spec(ColNames(), res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end
function try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_col)}, k::Spec, v)
	if is_table_spec(k)
		# Replace the inner spec

		# Are these correct? I think so.
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_spec(Col(spec.args[2]), res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end





function project(onto, t::TableFunction, args...)
	# Project the column names
	colnames = create_project_spec(get_colnames(onto), args...)
	onto2 = create_spec(ColNamesTableFunction(t.f), onto.args...; onto.kwargs...)
	create_project_spec(onto2, fetched(colnames), args...) # Consider moving colnames to a kwarg of project
end

function project(onto, s::ColNamesTableFunction, colnames, args...)
	# Given the column names, project the columns
	cols = (name=>create_project_spec(get_col(onto, name), args...) for name in colnames)
	create_table_impl_spec(cols...)
end



# --- Old ---

# dataframe_impl_spec(args::Pair...) = create_spec(DataFrame, args...; __version=v"0.0.1")
# dataframe_pr(action, args...) = dataframe_impl_spec(action.(args)...)
# dataframe_pr_spec(args...) = create_spec(Projectable(dataframe_pr), args...)



# function load_csv end # Defined in CSV package extension
# function get_csv_columns_pr end # Defined in CSV package extension
# function get_csv_columns_pr_spec end # Defined in CSV package extension

# # Could we do this with OncePerProcess instead? I'm not getting that to work for whatever reason.
# function __init__()
# 	Base.Experimental.register_error_hint(MethodError) do io, e, argtypes, kwargs
# 		if e.f == Jobs.load_csv && length(argtypes)==1
# 			printstyled(io, "\n`load_csv`"; color=:yellow)
# 			print(io, " is only available after loading ")
# 			printstyled(io, "`CSV.jl`"; color=:yellow)
# 			print(io, " (try running ")
# 			printstyled(io, "`using CSV`"; color=:yellow)
# 			print(io, ").")
# 		end
# 	end
# end


# get_columns_impl(df, columns...) = select(df, collect(columns); copycols=false)
# get_columns_impl_spec(df, columns) =
# 	create_spec(get_columns_impl, df, columns...; __version=v"0.1.0")
# get_columns_pr(action, df, columns...) =
# 	get_columns_impl_spec(action(df), action(columns)...)
# get_columns_pr_spec(df, columns...) =
# 	create_spec(Projectable(get_columns_pr), df, columns...) # General case


# function _subset_column_pairs(subset_names, column_pairs)
# 	ind = indexin(subset_names, first.(column_pairs))
# 	any(isnothing, ind) && throw(KeyError(subset_names[findfirst(isnothing,ind)]))
# 	column_pairs[ind]
# end

# function _subset_column_names(subset_names, column_names)
# 	ind = indexin(subset_names, column_names)
# 	any(isnothing, ind) && throw(KeyError(subset_names[findfirst(isnothing,ind)]))
# 	column_names[ind]
# end


# # TODO: This got a bit complicated, can we get around that?
# # TODO: Unit test all cases
# # TODO: Nested get_columns(get_columns(df,"id","DoesNotExist"), "id") will work. Is that bad? Or good? Or acceptable? Should we error because "DoesNotExist" doesn't exist? Can we know?
# function get_columns(df, columns...)
# 	if df isa Spec
# 		if df.f == Projectable(load_csv)
# 			return get_csv_columns_pr_spec(df.args[1], columns...; df.kwargs...)
# 		elseif df.f == Projectable(dataframe_pr) # DataFrame (Projectable)
# 			# Keep only chosen columns
# 			columns = _subset_column_pairs(columns, df.args) # This currently assumes the names are given as Strings (in both columns and df.args). Should we allow Specs returning strings as well?
# 			return dataframe_pr_spec(columns...)
# 		elseif df.f == DataFrame # DataFrame (Not Projectable)
# 			# Keep only chosen columns
# 			columns = _subset_column_pairs(columns, df.args) # This currently assumes the names are given as Strings (in both columns and df.args). Should we allow Specs returning strings as well?
# 			return dataframe_impl_spec(columns...)
# 		elseif df.f == Projectable(get_columns_pr) # nested get_columns (Projectable)
# 			column_names = _subset_column_names(columns, df.args[2:end])
# 			return get_columns_pr_spec(df.args[1], column_names...)
# 		elseif df.f == Projectable(get_csv_columns_pr) # get_columns(get_csv_columns) (Projectable)
# 			column_names = _subset_column_names(columns, df.args[2:end])
# 			return get_csv_columns_pr_spec(df.args[1], column_names...; df.kwargs...)
# 		elseif !(df.f isa Projectable) # General case (Not Projectable)
# 			return get_columns_impl_spec(df, columns...)
# 		end
# 		# Fallthrough if df.f is a Projectable
# 	end
# 	return get_columns_pr_spec(df, columns...) # General case
# end

# get_columns_spec(df, column1, columns...) =
# 	create_spec(Preprocess(get_columns), df, column1, columns...)

# function Jobs.get_columns(df, column1, columns...)
# 	Job(get_columns_spec(df, column1, columns...))
# end
