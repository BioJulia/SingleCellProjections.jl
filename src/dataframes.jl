dataframe_impl_spec(args::Pair...) = create_spec(DataFrame, args...; __version=v"0.0.1")
dataframe_pr(action, args...) = dataframe_impl_spec(action.(args)...)
dataframe_pr_spec(args...) = create_spec(Projectable(dataframe_pr), args...)



function load_csv end # Defined in CSV package extension
function get_csv_columns_pr end # Defined in CSV package extension

# Could we do this with OncePerProcess instead? I'm not getting that to work for whatever reason.
function __init__()
	Base.Experimental.register_error_hint(MethodError) do io, e, argtypes, kwargs
		if e.f == Jobs.load_csv && length(argtypes)==1
			printstyled(io, "\n`load_csv`"; color=:yellow)
			print(io, " is only available after loading ")
			printstyled(io, "`CSV.jl`"; color=:yellow)
			print(io, " (try running ")
			printstyled(io, "`using CSV`"; color=:yellow)
			print(io, ").")
		end
	end
end


get_columns_impl(df, columns...) = select(df, collect(columns); copycols=false)
get_columns_impl_spec(df, columns) =
	create_spec(get_columns_impl, df, columns...; __version=v"0.1.0")
get_columns_pr(action, df, columns...) =
	get_columns_impl_spec(action(df), action(columns)...)


function _subset_column_pairs(column_names, column_pairs)
	ind = indexin(column_names, first.(column_pairs))
	any(isnothing, ind) && throw(KeyError(column_names[findfirst(isnothing,ind)]))
	column_pairs[ind]
end


function get_columns(df, columns...)
	if df isa Spec && df.f == Projectable(load_csv)
		create_spec(Projectable(get_csv_columns_pr), df.args[1], columns...; df.kwargs...)
	elseif df isa Spec && df.f == Projectable(dataframe_pr) # DataFrame (Projectable)
		# Keep only chosen columns
		columns = _subset_column_pairs(columns, df.args) # This currently assumes the names are given as Strings (in both columns and df.args). Should we allow Specs returning strings as well?
		dataframe_pr_spec(columns...)
	elseif df isa Spec && df.f == DataFrame # DataFrame (Not Projectable)
		# Keep only chosen columns
		columns = _subset_column_pairs(columns, df.args) # This currently assumes the names are given as Strings (in both columns and df.args). Should we allow Specs returning strings as well?
		dataframe_impl_spec(columns...)
	elseif df isa Spec && !(df.f isa Projectable) # General case (Not Projectable)
		get_columns_impl_spec(df, columns...)
	else
		create_spec(Projectable(get_columns_pr), df, columns...) # General case
	end
end

function Jobs.get_columns(df, column1, columns...)
	Job(create_spec(Preprocess(get_columns), df, column1, columns...))
end
