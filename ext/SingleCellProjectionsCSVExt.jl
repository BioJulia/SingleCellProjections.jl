module SingleCellProjectionsCSVExt

using ReproducibleJobs
using ReproducibleJobs: create_spec, cached, Preprocessing
using SingleCellProjections
using SingleCellProjections: table_to_compound_result, table_from_compound_result

using DataFrames
using CSV


# TODO: Move some things into SingleCellProjectionsCore?

# _auto_delim(fp) = endswith(fp, r"\.(txt|tsv)(.gz)?"i) ? '\t' : ','

function parse_csv_impl(filepath; kwargs...)
	@assert filepath isa ReproducibleJobs.ChecksummedFilePath
	filepath = string(filepath)
	df = CSV.read(filepath, DataFrame; kwargs...)
	table_to_compound_result(df)
end
parse_csv_spec(filepath; kwargs...) =
	create_spec(parse_csv_impl, filepath; kwargs..., __version=v"0.0.3")




load_csv(::Preprocessing, filepath; kwargs...) =
	table_from_compound_result(parse_csv_spec(filepath; kwargs...))
function Jobs.load_csv(filepath::String; kwargs...)
	filepath_job = checksummedfilepath_job(filepath)
	Job(create_spec(Preprocess(load_csv), filepath_job; kwargs...))
end

end
