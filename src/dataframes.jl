dataframe_spec(args::Pair...) = create_spec(DataFrame, args...; __version=v"0.0.1")



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



function get_columns(df, columns...)
	if df isa Spec && df.f == Projectable(load_csv)
		create_spec(Projectable(get_csv_columns_pr), df.args[1], columns...; df.kwargs...)
	else
		error("Not implemented")
		# This should also return a Projectable
	end
end

function Jobs.get_columns(df, column1, columns...)
	Job(create_spec(Preprocess(get_columns), df, column1, columns...))
end
