module SingleCellProjectionsCSVExt

using ReproducibleJobs
using ReproducibleJobs: create_spec, cached, CompoundResult
using SingleCellProjections
using SingleCellProjections: Projectable, Action, table_from_compound_result
# using SingleCellProjections: Projectable, Action, TableFunction, TableField, ColNames, Col, create_table_impl # old

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
parse_csv_spec(filepath; delim) =
	create_spec(parse_csv_impl, filepath; delim, __version=v"0.0.3")


# function load_csv_from_colnames(colnames, parsed)
# 	cols = (name=>cached(parsed, name) for name in colnames)
# 	create_table_impl_spec(cols...)
# end

# function load_csv(filepath; delim=_auto_delim(filepath), kwargs...)
# 	parsed = parse_csv_spec(filepath; delim)
# 	colnames = fetched(cached(parsed; return_keys=true))
# 	create_spec(Preprocess(load_csv_from_colnames), colnames, parsed)
# end
function load_csv(filepath; delim=_auto_delim(filepath), kwargs...)
	parsed = parse_csv_spec(filepath; delim)
	table_from_compound_result(parsed)
end


function Jobs.load_csv(filepath::String; kwargs...)
	filepath_job = checksummedfilepath_job(filepath)
	Job(create_spec(Preprocess(load_csv), filepath_job; kwargs...))
end


# --- Old ---

# _auto_delim(fp) = lowercase(splitext(string(fp))[2]) in (".tsv",".txt") ? '\t' : ','

# function parse_csv_impl(filepath; delim)
# 	@assert filepath isa ReproducibleJobs.ChecksummedFilePath
# 	filepath = string(filepath)
# 	df = CSV.read(filepath, DataFrame; delim)
# 	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(df))])
# end
# parse_csv_spec(filepath; delim) =
# 	create_spec(parse_csv_impl, filepath; delim, __version=v"0.0.3")


# function load_csv_keys_pr(::Action, filepath; delim)
# 	# No action applied, to change what is loaded, replacements should be used (e.g. base_csv=>projected_csv) in project.
# 	parse_job = parse_csv_spec(filepath; delim)
# 	cached(parse_job; return_keys=true)
# end
# function load_csv_column_pr(::Action, filepath, column; delim)
# 	# No action applied, to change what is loaded, replacements should be used (e.g. base_csv=>projected_csv) in project.
# 	parse_job = parse_csv_spec(filepath; delim)
# 	cached(parse_job, column)
# end


# load_csv(::ColNames, filepath; delim=_auto_delim(filepath)) =
# 	create_spec(Projectable(load_csv_keys_pr), filepath; delim)
# load_csv(c::Col, filepath; delim=_auto_delim(filepath)) =
# 	create_spec(Projectable(load_csv_column_pr), filepath, c.name; delim)

# function Jobs.load_csv(path::String; kwargs...)
# 	filepath_job = checksummedfilepath_job(path)
# 	Job(create_spec(TableFunction(load_csv), filepath_job; kwargs...))
# end



end
