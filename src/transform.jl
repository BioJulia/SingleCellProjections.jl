"""
    SCP.logtransform([T=Float64,] counts; scale_factor=10_000) -> Job

Apply log transformation: `log(1 + x * scale_factor / total_counts)`. Returns a
`DataMatrix` with the transformed matrix. The element type of the resulting matrix is `T`.

(TODO: Add example.)

See also [`sctransform`](@ref), [`normalize_matrix`](@ref).
"""
function logtransform(T::DataType, counts; scale_factor=10_000)
	create_job(DataMatrixFunction(Impl.logtransform), T, counts; scale_factor)
end
logtransform(counts; kwargs...) = logtransform(Float64, counts; kwargs...)


"""
    SCP.sctransform([T=Float64,] counts; kwargs...) -> Job

Apply SCTransform (variance-stabilizing transformation) to raw count data. Returns a
`DataMatrix` with the transformed matrix. The element type of the resulting matrix is `T`.

Keyword arguments:
- `var_filter` — filter variables used for parameter estimation (default `:`).
- `min_cells` — minimum number of cells with nonzero counts for a variable to be included (default `5`).
- `annotate` — if `true`, add SCTransform parameters to `var` annotations.

# Examples

SCTransform a `counts` data matrix.
```julia
julia> SCP.sctransform(counts)
```

See also [`logtransform`](@ref), [`normalize_matrix`](@ref).
"""
function sctransform(T::DataType, counts; kwargs...)
	check_kwargs(kwargs, :var_filter, :min_cells, :annotate, :clip, :rtol, :atol)
	create_job(DataMatrixFunction(Impl.sctransform), T, counts; kwargs...)
end
sctransform(counts; kwargs...) = sctransform(Float64, counts; kwargs...)


"""
    SCP.tf_idf_transform([T=Float64,] counts; scale_factor=10_000, annotate=false, kwargs...) -> Job

Apply the TF-IDF (term frequency-inverse document frequency) transform to raw count data.
Returns a `DataMatrix` with the transformed matrix. The element type of the resulting matrix is `T`.

The transform is `log(1 + scale_factor * tf * idf)`, where the term frequency is
`tf = counts ./ max.(1, sum(counts; dims=1))` and the inverse document frequency is
`idf = nobs ./ max.(1, sum(counts; dims=2))`.

`idf` is estimated from `counts` and stored in the model, so that projecting onto another dataset
reuses it (remapping to the projected variables by ID) rather than recomputing.

Keyword arguments:
- `scale_factor` — term-frequency scale factor (default `10_000`).
- `annotate` — if `true`, add the `idf` vector as a `var` annotation.

See also [`logtransform`](@ref), [`sctransform`](@ref), [`normalize_matrix`](@ref).
"""
function tf_idf_transform(T::DataType, counts; scale_factor=10_000, kwargs...)
	check_kwargs(kwargs, :annotate)
	create_job(DataMatrixFunction(Impl.tf_idf_transform), T, counts; scale_factor, kwargs...)
end
tf_idf_transform(counts; kwargs...) = tf_idf_transform(Float64, counts; kwargs...)
