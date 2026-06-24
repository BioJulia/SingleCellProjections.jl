module CSVExt

using ReproducibleJobs
using ReproducibleJobs: create_job, cached, Preprocessing
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
parse_csv_job(filepath; kwargs...) =
	create_job(parse_csv_impl, filepath; kwargs..., __version=v"0.0.3")




load_csv(::Preprocessing, filepath; kwargs...) =
	table_from_compound_result(parse_csv_job(filepath; kwargs...))
"""
    Jobs.load_csv(filepath; kwargs...) -> Job

Load a CSV or TSV file as a table `Job`. The file path is automatically checksummed for
cache invalidation. Requires the `CSV` package to be loaded.

See also [`Jobs.load_counts`](@ref).
"""
function Jobs.load_csv(filepath::Union{String,TimestampedFilePath}; kwargs...)
	filepath_job = checksummedfilepath_job(filepath)
	create_job(Preprocess(load_csv), filepath_job; kwargs...)
end

end
