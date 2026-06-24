# low-level specs for loading counts
function load_var_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_features(filename, DataFrame)
end
load_var_job(filename) = create_job(load_var_impl, filename; __version=v"0.1.0")

function load_barcodes_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_barcodes(filename)
end
load_barcodes_job(filename; kwargs...) = create_job(load_barcodes_impl, filename; kwargs..., __version=v"0.1.0")


function vcat_tables(tables; kwargs...)
	# df = vcat(tables...; kwargs...) # TODO: splat here or not?
	df = reduce(vcat, tables)
	table_to_compound_result(df)
end
vcat_tables_job(tables; kwargs...) = create_job(vcat_tables, tables; kwargs..., __version=v"0.1.0")


function combine_obs(::Preprocessing, filenames, sample_names)
	sample_barcodes_specs = load_barcodes_job.(filenames)
	sample_id_specs = combine_vectors_job.(sample_names, '_', sample_barcodes_specs)
	sample_obs_specs = create_table_job.("cell_id" .=> sample_id_specs,
	                                      "sample_name" .=> sample_names,
	                                      "barcode" .=> sample_barcodes_specs)
	combined = vcat_tables_job(sample_obs_specs)

	table_from_compound_result(combined, ["cell_id", "sample_name", "barcode"])
end

combine_obs_job(filenames, sample_names) =
	create_job(Preprocess(combine_obs), filenames, sample_names)


function combine_var_impl(vars; kwargs...)
	df = SCPCore.combine_var(vars; kwargs...)
	table_to_compound_result(df)
end


function combine_var(::Preprocessing, vars; kwargs...)
	combined = create_job(combine_var_impl, vars; kwargs..., __version=v"0.1.1")
	table_from_compound_result(combined)
end


combine_var_job(vars; kwargs...) =
	create_job(Preprocess(combine_var), vars; kwargs...)


sample_var_indices_job(sample_var, var; kwargs...) =
	cached(create_job(SCPCore.sample_var_indices, sample_var, var; kwargs..., __version=v"0.1.0"))

function load_sample_matrix_metadata_impl(filename, var_ind)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SCPCore.load_sample_matrix_metadata(filename, var_ind)
end
load_sample_matrix_metadata_job(filename, var_ind; kwargs...) =
	cached(create_job(load_sample_matrix_metadata_impl, filename, var_ind; kwargs..., __version=v"0.1.0"))



# Old unblocked version
# function load_hcat_sample_matrices_impl(filenames, matrix_metadatas, var_inds; Tv=Int, Ti=Int32)
# 	@assert all(x->x isa ChecksummedFilePath, filenames)
# 	filenames = string.(filenames)
# 	SCPCore.load_hcat_sample_matrices(Tv, Ti, filenames, matrix_metadatas, var_inds)
# end
# function load_hcat_sample_matrices_job(filenames, matrix_metadatas, var_inds; Tv=Int, Ti=Int32)
# 	kwargs = (;)
# 	if Tv != Int || Ti != Int32
# 		kwargs = (; Tv, Ti)
# 	end
# 	create_job(load_hcat_sample_matrices_impl, filenames, matrix_metadatas, var_inds; kwargs..., __version=v"0.1.0")
# end

# function load_counts(f::Union{Mat,Var}, filename_specs; sample_names, prefilter, extra_id_cols, kwargs...)
# 	sample_var_specs = load_var_job.(filename_specs)
# 	var_job = combine_var_job(sample_var_specs; prefilter, extra_id_cols)

# 	if f isa Var
# 		return var_job
# 	else # if f isa Mat
# 		var_ind_specs = prefetched.(sample_var_indices_job.(sample_var_specs, var_job; extra_id_cols))
# 		metadata_specs = prefetched.(load_sample_matrix_metadata_job.(filename_specs, var_ind_specs))
# 		return load_hcat_sample_matrices_job(filename_specs, metadata_specs, var_ind_specs; kwargs...)
# 	end
# end
# load_counts(::Obs, filename_specs; sample_names, prefilter, extra_id_cols, kwargs...) =
# 	combine_obs_job(filename_specs, sample_names)

# function Jobs.load_counts(filenames;
#                           sample_names,
#                           prefilter = "feature_type"=>isequal("Gene Expression"),
#                           extra_id_cols = "feature_type",
#                           kwargs...)
# 	filenames isa AbstractArray || (filenames = [filenames])
# 	sample_names isa AbstractArray || (sample_names = [sample_names])

# 	# TODO: Support .mtx.gz, with the added complication that we need to find matching
# 	#       feature/barcode files already here so that they can be checksummed too.
# 	@assert all(x->lowercase(splitext(x)[2])==".h5", filenames) "Only 10x .h5 files are currently supported"

# 	filename_specs = checksummedfilepath_job.(filenames)
# 	create_job(DataMatrixFunction(load_counts), filename_specs; sample_names, prefilter, extra_id_cols, kwargs...)
# end



# Blocked version
function load_sample_matrix_impl(filename::ChecksummedFilePath, var_ind; Tv=Int, Ti=Int32, row_block_size, col_block_size)
	X = SCPCore.load_sample_matrix(Tv, Ti, string(filename), var_ind)
	SCPCore.blockify(X; row_block_size, col_block_size)
end
load_sample_matrix_job(filename, var_ind; row_block_size=1024, col_block_size=1024, kwargs...) =
	create_job(load_sample_matrix_impl, filename, var_ind; row_block_size, col_block_size, kwargs..., __version=v"0.1.0")


function metadata_to_hblock_ranges(metadata::AbstractVector{Tuple{Int,Int,Int}})
	Ns = getindex.(metadata, 2)
	SCPCore.block_sizes_to_ranges(Ns)
end
metadata_to_hblock_ranges_job(metadata) =
	create_job(metadata_to_hblock_ranges, metadata; __version=v"0.0.1")


function load_counts(f::Union{Mat,Var}, filename_specs; sample_names, prefilter, extra_id_cols, kwargs...)
	sample_var_specs = load_var_job.(filename_specs)
	var_job = combine_var_job(sample_var_specs; prefilter, extra_id_cols)

	if f isa Var
		return var_job
	else # if f isa Mat
		var_ind_specs = prefetched.(sample_var_indices_job.(sample_var_specs, var_job; extra_id_cols))
		sample_specs = load_sample_matrix_job.(filename_specs, var_ind_specs)

		if length(sample_specs)	== 1
			return only(sample_specs)
		else
			metadata_specs = load_sample_matrix_metadata_job.(filename_specs, var_ind_specs)
			ranges = fetched(metadata_to_hblock_ranges_job(vcat_job(metadata_specs)))
			return hblock_job(sample_specs, ranges)
		end
	end
end
load_counts(::Obs, filename_specs; sample_names, prefilter, extra_id_cols, kwargs...) =
	combine_obs_job(filename_specs, sample_names)

"""
    Jobs.load_counts(filenames; sample_names, prefilter="feature_type"=>isequal("Gene Expression"), extra_id_cols="feature_type", kwargs...) -> Job

Load raw count matrices from one or more 10x HDF5 (.h5) files. Returns a `Job` whose
result is a `DataMatrix` with genes as variables and cells as observations.

* `sample_names` is required and assigns a name to each sample.
* `prefilter` selects which features to keep (defaults to Gene Expression only).
* `extra_id_cols` specifies additional columns used (together with the first column) to uniquely identify variables when merging samples. Variables with matching ID columns are combined.

# Examples

```julia
julia> Jobs.load_counts("SampleA.h5"; sample_names="SampleA")

julia> Jobs.load_counts(["SampleA.h5", "SampleB.h5"]; sample_names=["SampleA","SampleB"])
```


See also `Jobs.load_csv`.
"""
function Jobs.load_counts(filenames;
                          sample_names,
                          prefilter = "feature_type"=>isequal("Gene Expression"),
                          extra_id_cols = "feature_type", # TODO: Remove this default value?
                          kwargs...)
	filenames isa AbstractArray || (filenames = [filenames])
	sample_names isa AbstractArray || (sample_names = [sample_names])

	# TODO: Support .mtx.gz, with the added complication that we need to find matching
	#       feature/barcode files already here so that they can be checksummed too.
	@assert all(x->lowercase(splitext(x)[2])==".h5", filenames) "Only 10x .h5 files are currently supported"

	filename_specs = checksummedfilepath_job.(filenames)
	create_job(DataMatrixFunction(load_counts), filename_specs; sample_names, prefilter, extra_id_cols, kwargs...)
end
