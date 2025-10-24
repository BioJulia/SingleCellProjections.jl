module SingleCellProjectionsCSVExt

using ReproducibleJobs
using ReproducibleJobs: create_spec, cached, CompoundResult
using SingleCellProjections
using SingleCellProjections: Projectable, Action, dataframe_impl_spec

using DataFrames
using CSV


# TODO: Move some things into SingleCellProjectionsCore?


_auto_delim(fp) = lowercase(splitext(string(fp))[2]) in (".tsv",".txt") ? '\t' : ','

function parse_csv_impl(filepath; delim)
	@assert filepath isa ReproducibleJobs.ChecksummedFilePath
	filepath = string(filepath)
	df = CSV.read(filepath, DataFrame; delim)
	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(df))])
end


function get_csv_columns_pre(parse_job, columns)
	# TODO: Can we avoid this somehow? It is not supposed to be like this.
	if columns isa ReproducibleJobs.ReadOnly
		columns = columns.value
	end

	col_jobs = (col=>cached(parse_job, col) for col in columns)
	dataframe_impl_spec(col_jobs...)
end


function get_csv_columns_spec(filepath, columns...; delim=_auto_delim(filepath))
	parse_job = create_spec(parse_csv_impl, filepath; delim, __version=v"0.0.3")
	if isempty(columns)
		columns = fetched(cached(parse_job; return_keys=true))
	else
		columns = collect(columns)
	end
	create_spec(Preprocess(get_csv_columns_pre), parse_job, columns)
end



# used by SingleCellProjections.get_columns
SingleCellProjections.get_csv_columns_pr(::Action, filepath, columns...; kwargs...) =
	get_csv_columns_spec(filepath, columns...; kwargs...)


SingleCellProjections.load_csv(::Action, filepath; kwargs...) =
	get_csv_columns_spec(filepath; kwargs...)

# TODO: kwarg for selecting id column somehow. Possibly even for creating one from multiple columns by concatenation?
# Or do those things as a separate job?
function Jobs.load_csv(path::String; kwargs...)
	filepath_job = checksummedfilepath_job(path)
	Job(create_spec(Projectable(SingleCellProjections.load_csv), filepath_job; kwargs...))
end



end
