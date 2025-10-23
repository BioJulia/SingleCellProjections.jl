module SingleCellProjectionsCSVExt

using ReproducibleJobs
using ReproducibleJobs: create_spec, cached, CompoundResult
using SingleCellProjections
using SingleCellProjections: Projectable, Action, dataframe_spec

using DataFrames
using CSV


# TODO: Move some things into SingleCellProjectionsCore?


_auto_delim(fp) = lowercase(splitext(string(fp))[2]) in (".tsv",".txt") ? '\t' : ','

function parse_csv_impl(filepath; delim)
	@show typeof(filepath)
	@show typeof(delim)

	@assert filepath isa ReproducibleJobs.ChecksummedFilePath
	filepath = string(filepath)
	df = CSV.read(filepath, DataFrame; delim)
	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(df))])
end
function parse_csv_pr(::Action, filepath; delim=nothing)
	delim = @something _auto_delim(filepath)
	create_spec(parse_csv_impl, filepath; delim, __version=v"0.0.3")
end


function load_dataframe_columns(parse_job, keys)
	col_jobs = (k=>cached(parse_job, k) for k in keys)
	dataframe_spec(col_jobs...)
end


function load_parsed_csv(fp; kwargs...)
	parse_job = create_spec(Projectable(parse_csv_pr), fp; kwargs...)
	keys_job = fetched(cached(parse_job; return_keys=true))
	create_spec(Preprocess(load_dataframe_columns), parse_job, keys_job)
end


function Jobs.load_csv(path::String; kwargs...)
	filepath_job = checksummedfilepath_job(path)
	Job(create_spec(Preprocess(load_parsed_csv), filepath_job; kwargs...))
end




end
