abstract type TableField end # Find a better name?
struct ColNames <: TableField end
struct Col <: TableField
	name::String
end



create_table_impl(args::Pair...) = DataFrame(args...)
create_table_pr(action::Action, args::Pair...) = create_spec(create_table_impl, action(args)...; __version=v"0.0.1")
create_table_spec(args::Pair...) = create_spec(Projectable(create_table_pr), args...)






# Do we need this?
# is_table_spec(::Any) = false
# function is_table_spec(sa::SpecArgs)
# 	f = sa.f
# 	f isa TableFunction && return true
# 	f == create_table_impl && return true
#   more?
# 	if f == project
# 		onto = sa.args[1]
# 		return is_table_spec(onto)
# 	end
# 	# TODO: Are there more cases that should return true?
# 	return false
# end
# is_table_spec(spec::Spec) = is_table_spec(spec.ro.value)



function _setup_table(f::F, col_names, args...; kwargs...) where F
	columns = (name=>f(Col(name), args...; kwargs...) for name in col_names)
	create_table_spec(columns...)
end


# This evaluates the TableFunction
function (d::TableFunction{F})(args...; kwargs...) where F
	col_names = d.f(ColNames(), args...; kwargs...) # can be a spec or just a list of names

	# We need to preprocess once to fetch the col_names.
	# This also means that col_names can be Projected.
	create_spec(SetupTable(d.f), fetched(col_names), args...; kwargs...)

	# TODO: Should we do this? It's just a small shortcut for when col_names are not a Spec. But it should work fine with projections too.
	# if col_names isa Union{Spec,Job}
	# 	# We need to preprocess once to fetch the col_names
	# 	create_spec(SetupTable(d.f), fetched(col_names), args...; kwargs...)
	# else
	# 	# col_names are not a spec, just setup directly
	# 	_setup_table(d.f, col_names, args...; kwargs...)
	# end
end


function (d::SetupTable{F})(col_names, args...; kwargs...) where F
	_setup_table(d.f, col_names, args...; kwargs...)
end




# function project(onto, t::TableFunction, args...)
# 	create_table_spec(...)
# end

# function project(onto, s::SetupTable, args...)
# 	# project names and columns...
# 	create_table_spec(...)
# end



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
