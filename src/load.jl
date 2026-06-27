"""
    SCP.load_counts(filenames; sample_names, prefilter="feature_type"=>isequal("Gene Expression"), extra_id_cols="feature_type", kwargs...) -> Job

Load raw count matrices from one or more 10x HDF5 (.h5) files. Returns a `Job` whose
result is a `DataMatrix` with genes as variables and cells as observations.

* `sample_names` is required and assigns a name to each sample.
* `prefilter` selects which features to keep (defaults to Gene Expression only).
* `extra_id_cols` specifies additional columns used (together with the first column) to uniquely identify variables when merging samples. Variables with matching ID columns are combined.

# Examples

Load a single sample:
```julia
julia> SCP.load_counts("SampleA.h5"; sample_names="SampleA")
```

Load multiple samples:
```julia
julia> SCP.load_counts(["SampleA.h5", "SampleB.h5"]; sample_names=["SampleA","SampleB"])
```


See also [`load_csv`](@ref).
"""
function load_counts(filenames;
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
	create_job(Impl.DataMatrixFunction(Impl.load_counts), filename_specs; sample_names, prefilter, extra_id_cols, kwargs...)
end
