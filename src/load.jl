# low-level specs for loading counts
function load_var_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_features(filename, DataFrame)
end
load_var_spec(filename) = create_spec(load_var_impl, filename; __version=v"0.1.0")

function load_barcodes_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_barcodes(filename)
end
load_barcodes_spec(filename; kwargs...) = create_spec(load_barcodes_impl, filename; kwargs..., __version=v"0.1.0")


# function combine_obs_impl(obs, sample_names; kwargs...)
# 	df = SCPCore.combine_obs(obs, sample_names; kwargs...)
# 	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(df))])
# end
# combine_obs_impl_spec(obs, sample_names; id_col="cell_id", id_delim='_', sample_name_col="sample_name", kwargs...) =
# 	create_spec(combine_obs_impl, obs, sample_names; id_col, id_delim, sample_name_col, kwargs..., __version=v"0.3.0")

# function combine_obs(::ColNames, obs, sample_names; kwargs...)
# 	combine_job = combine_obs_impl_spec(obs, sample_names; kwargs...)
# 	cached(combine_job; return_keys=true)
# end
# function combine_obs(c::Col, obs, sample_names; kwargs...)
# 	combine_job = combine_obs_impl_spec(obs, sample_names; kwargs...)
# 	cached(combine_job, c.name)
# end

# combine_obs_spec(obs, sample_names; kwargs...) =
# 	create_spec(TableFunction(combine_obs), obs, sample_names; kwargs...)



# # --- TESTING -------------------------------------------------------
# # Find a better name? It just broadcasts string across the args.
# combine_cols_impl(args...) = string.(args...)
# combine_cols_spec(args...) = create_spec(combine_cols_impl, args...; __version=v"0.1.0")


# function vcat_tables(tables...; kwargs...)
# 	df = vcat(tables...; kwargs...)
# 	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(df))])
# end
# vcat_tables_spec(tables...; kwargs...) = create_spec(vcat_tables, tables...; kwargs..., __version=v"0.1.0")


# combine_obs_table(::ColNames, combined) = cached(combined; return_keys=true)
# combine_obs_table(c::Col, combined) = cached(combined, c.name)

# function combine_obs(filenames, sample_names)
# 	if sample_names isa ReadOnly # Can we avoid this?
# 		sample_names = sample_names.value
# 	end

# 	sample_barcodes_specs = load_barcodes_spec.(filenames)
# 	sample_id_specs = combine_cols_spec.(sample_names, '_', sample_barcodes_specs)
# 	# sample_obs_specs = create_table_impl_spec.("cell_id".=>sample_id_specs, "barcode" .=> sample_barcodes_specs)
# 	sample_obs_specs = create_table_impl_spec.("cell_id".=>sample_id_specs, "sample_name".=>sample_names, "barcode" .=> sample_barcodes_specs)
# 	combined = vcat_tables_spec(sample_obs_specs...)
# 	create_spec(TableFunction(combine_obs_table), combined)
# end

# combine_obs_spec(filenames, sample_names) =
# 	create_spec(Preprocess(combine_obs), filenames, sample_names)
# # --- TESTING -------------------------------------------------------


# Find a better name? It just broadcasts string across the args.
combine_cols_impl(args...) = string.(args...)
combine_cols_spec(args...) = create_spec(combine_cols_impl, args...; __version=v"0.1.0")


function vcat_tables(tables...; kwargs...)
	df = vcat(tables...; kwargs...)
	CompoundResult(Pair{String,Any}[string(name)=>col for (name,col) in pairs(eachcol(df))])
end
vcat_tables_spec(tables...; kwargs...) = create_spec(vcat_tables, tables...; kwargs..., __version=v"0.1.0")


function combine_obs(filenames, sample_names)
	if sample_names isa ReadOnly # Can we avoid this?
		sample_names = sample_names.value
	end
	sample_barcodes_specs = load_barcodes_spec.(filenames)
	sample_id_specs = combine_cols_spec.(sample_names, '_', sample_barcodes_specs)
	sample_obs_specs = create_table_impl_spec.("cell_id".=>sample_id_specs, "sample_name".=>sample_names, "barcode" .=> sample_barcodes_specs)
	combined = vcat_tables_spec(sample_obs_specs...)
	create_table_impl_spec("cell_id"=>cached(combined,"cell_id"), "sample_name"=>cached(combined,"sample_name"), "barcode"=>cached(combined,"barcode"))
end

combine_obs_spec(filenames, sample_names) =
	create_spec(Preprocess(combine_obs), filenames, sample_names)



combine_var_spec(vars; kwargs...) = cached(create_spec(SCPCore.combine_var, vars; kwargs..., __version=v"0.1.0")) # TODO: Do we want cache here? Probably.

sample_var_indices_spec(sample_var, var; kwargs...) =
	cached(create_spec(SCPCore.sample_var_indices, sample_var, var; kwargs..., __version=v"0.1.0"))

function load_sample_matrix_metadata_impl(filename, var_ind)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SCPCore.load_sample_matrix_metadata(filename, var_ind)
end
load_sample_matrix_metadata_spec(filename, var_ind; kwargs...) =
	cached(create_spec(load_sample_matrix_metadata_impl, filename, var_ind; kwargs..., __version=v"0.1.0"))

function load_hcat_sample_matrices_impl(filenames, args...; kwargs...)
	@assert all(x->x isa ChecksummedFilePath, filenames)
	filenames = string.(filenames)
	SCPCore.load_hcat_sample_matrices(filenames, args...; kwargs...)
end
load_hcat_sample_matrices_spec(filenames, matrix_metadatas, var_inds; kwargs...) =
	create_spec(load_hcat_sample_matrices_impl, filenames, matrix_metadatas, var_inds; kwargs..., __version=v"0.1.0")




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
	# sample_barcodes_specs = load_barcodes_spec.(filename_specs)
	# # sample_obs_specs = create_table_impl_spec.("barcode" .=> sample_barcodes_specs)
	# sample_id_specs = combine_cols_spec.(sample_names, '_' sample_barcodes_specs)
	# sample_obs_specs = create_table_impl_spec.("barcode" .=> sample_barcodes_specs, "cell_id".=>sample_id_specs)
	# combined = vcat_tables(sample_obs_specs...)

	# combine_obs_spec(sample_obs_specs, sample_names)

	combine_obs_spec(filename_specs, sample_names)
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
