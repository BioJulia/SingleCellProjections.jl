# low-level specs for loading counts
function load_var_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_features(filename, DataFrame)
end
load_var_spec(filename) = create_spec(load_var_impl, filename; __use_cache=false, __version=v"0.1.0")

function load_obs_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_barcodes(filename, DataFrame)
end
load_obs_spec(filename; kwargs...) = create_spec(load_obs_impl, filename; __use_cache=false, kwargs..., __version=v"0.1.0")

combine_obs_spec(obs, sample_names; id_col="cell_id", id_delim='_', sample_name_col="sample_name", kwargs...) =
	create_spec(SCPCore.combine_obs, obs, sample_names; id_col, id_delim, sample_name_col, __use_cache=true, kwargs..., __version=v"0.2.0") # TODO: Do we want __use_cache=true here? Probably.

combine_var_spec(vars; kwargs...) = create_spec(SCPCore.combine_var, vars; __use_cache=true, kwargs..., __version=v"0.1.0") # TODO: Do we want __use_cache=true here? Probably.

sample_var_indices_spec(sample_var, var; kwargs...) =
	create_spec(SCPCore.sample_var_indices, sample_var, var; __use_cache=true, kwargs..., __version=v"0.1.0")

function load_sample_matrix_metadata_impl(filename, var_ind)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SCPCore.load_sample_matrix_metadata(filename, var_ind)
end
load_sample_matrix_metadata_spec(filename, var_ind; kwargs...) =
	create_spec(load_sample_matrix_metadata_impl, filename, var_ind; __use_cache=true, kwargs..., __version=v"0.1.0")

function load_hcat_sample_matrices_impl(filenames, args...; kwargs...)
	@assert all(x->x isa ChecksummedFilePath, filenames)
	filenames = string.(filenames)
	SCPCore.load_hcat_sample_matrices(filenames, args...; kwargs...)
end
load_hcat_sample_matrices_spec(filenames, matrix_metadatas, var_inds; kwargs...) =
	create_spec(load_hcat_sample_matrices_impl, filenames, matrix_metadatas, var_inds; __use_cache=false, kwargs..., __version=v"0.1.0")




function load_counts(f::Union{Mat,Var}, filename_specs; sample_names, prefilter, extra_id_cols)
	sample_var_specs = load_var_spec.(filename_specs)
	var_spec = combine_var_spec(sample_var_specs; prefilter, extra_id_cols)

	if f isa Var
		return var_spec
	else # if f isa Mat
		var_ind_specs = prefetched.(sample_var_indices_spec.(sample_var_specs, var_spec; extra_id_cols))
		metadata_specs = prefetched.(load_sample_matrix_metadata_spec.(filename_specs, var_ind_specs))
		return load_hcat_sample_matrices_spec(filename_specs, metadata_specs, var_ind_specs)
	end
end
function load_counts(::Obs, filename_specs; sample_names, prefilter, extra_id_cols)
	sample_obs_specs = load_obs_spec.(filename_specs)
	combine_obs_spec(sample_obs_specs, sample_names)
end


function Jobs.load_counts(filenames;
                          sample_names,
                          prefilter = "feature_type"=>isequal("Gene Expression"),
                          extra_id_cols = "feature_type",
                          kwargs...)
	filenames isa AbstractArray || (filenames = [filenames])
	sample_names isa AbstractArray || (sample_names = [sample_names])

	# TODO: Support .mtx.gz, with the added complication that we need to find matching
	#       feature/barcode files already here so that they can be checksummed too.
	@assert all(x->lowercase(splitext(x)[2])==".h5", filenames) "Only 10x .h5 files are currently supported"

	filename_jobs = checksummedfilepath_job.(filenames)
	Job(create_spec(DataMatrixFunction(load_counts), filename_jobs; sample_names, prefilter, extra_id_cols, kwargs...))
end
