_is_h5(filename) = lowercase(splitext(filename)[2]) == ".h5"

# `nothing` -> guess a sibling file for each input; otherwise normalize to a vector matching `filenames`.
function _component_filenames(given, filenames, guess)
	given === nothing && return guess.(filenames)
	given isa AbstractArray || (given = [given])
	length(given) == length(filenames) ||
		throw(ArgumentError("Expected $(length(filenames)) filename(s), got $(length(given))."))
	given
end

"""
    SCP.load_counts(filenames; sample_names, feature_filenames=nothing, barcode_filenames=nothing, prefilter="feature_type"=>isequal("Gene Expression"), extra_id_cols="feature_type", kwargs...) -> Job

Load raw count matrices from one or more 10x files. Returns a `Job` whose result is a
`DataMatrix` with genes as variables and cells as observations.

Each file can be a 10x HDF5 (`.h5`) file, or a CellRanger Matrix Market matrix
(`.mtx[.gz]`). For a `.mtx` file, the matching feature and barcode files are found in the same
folder (following the CellRanger naming convention), or can be given explicitly.

* `sample_names` is required and assigns a name to each sample.
* `feature_filenames` / `barcode_filenames` — explicit feature/barcode files (a single filename or a vector matching `filenames`). When `nothing` (default), they are guessed from each `.mtx` filename; for a `.h5` file the file itself is used.
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

Load from a Matrix Market file (features/barcodes found in the same folder):
```julia
julia> SCP.load_counts("matrix.mtx.gz"; sample_names="SampleA")
```


See also [`load_csv`](@ref).
"""
function load_counts(filenames;
                          sample_names,
                          feature_filenames = nothing,
                          barcode_filenames = nothing,
                          prefilter = "feature_type"=>isequal("Gene Expression"),
                          extra_id_cols = "feature_type", # TODO: Remove this default value?
                          kwargs...)
	filenames isa AbstractArray || (filenames = [filenames])
	sample_names isa AbstractArray || (sample_names = [sample_names])

	matrix_specs = checksummedfilepath_job.(filenames)

	# Legacy path: pure .h5 with no explicit component files - keep the spec (and hashes) unchanged.
	if feature_filenames === nothing && barcode_filenames === nothing && all(_is_h5, filenames)
		return create_job(DataMatrixFunction(Impl.load_counts), matrix_specs; sample_names, prefilter, extra_id_cols, kwargs...)
	end

	# General path: .mtx and/or explicit feature/barcode files. Guess the sibling files when not given
	# (for a .h5 input the guess returns the .h5 path itself).
	feature_files = _component_filenames(feature_filenames, filenames, SingleCell10x.guessfeaturefilename)
	barcode_files = _component_filenames(barcode_filenames, filenames, SingleCell10x.guessbarcodefilename)
	feature_specs = checksummedfilepath_job.(feature_files)
	barcode_specs = checksummedfilepath_job.(barcode_files)

	create_job(DataMatrixFunction(Impl.load_counts), matrix_specs, feature_specs, barcode_specs; sample_names, prefilter, extra_id_cols, kwargs...)
end
