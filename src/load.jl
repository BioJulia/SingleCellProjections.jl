_is_h5(filename) = lowercase(splitext(filename)[2]) == ".h5"

"""
    SCP.load_counts(filenames; sample_names, feature_filenames=nothing, barcode_filenames=nothing, prefilter="feature_type"=>isequal("Gene Expression"), extra_id_cols="feature_type") -> Job

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
                     extra_id_cols = "feature_type") # TODO: Remove this default value?
	filenames isa AbstractArray || (filenames = [filenames])
	sample_names isa AbstractArray || (sample_names = [sample_names])

	matrix_specs = checksummedfilepath_job.(filenames)

	extra_kwargs = (;)
	if !all(_is_h5, filenames) || feature_filenames !== nothing || barcode_filenames !== nothing
		# We need to specify feature and barcode filenames separately.

		feature_filenames = @something feature_filenames SingleCell10x.guessfeaturefilename.(filenames)
		feature_filenames isa AbstractArray || (feature_filenames = [feature_filenames])
		feature_specs = checksummedfilepath_job.(feature_filenames)

		barcode_filenames = @something barcode_filenames SingleCell10x.guessbarcodefilename.(filenames)
		barcode_filenames isa AbstractArray || (barcode_filenames = [barcode_filenames])
		barcode_specs = checksummedfilepath_job.(barcode_filenames)

		nm = length(matrix_specs)
		nf = length(feature_specs)
		nb = length(barcode_specs)

		nm != nf && throw(ArgumentError("The number of matrix filenames ($nm) and feature filenames ($nf) are not equal."))
		nm != nb && throw(ArgumentError("The number of matrix filenames ($nm) and barcode filenames ($nb) are not equal."))

		extra_kwargs = (; feature_specs, barcode_specs)
	end
	create_job(DataMatrixFunction(Impl.load_counts), matrix_specs; sample_names, prefilter, extra_id_cols, extra_kwargs...)
end
