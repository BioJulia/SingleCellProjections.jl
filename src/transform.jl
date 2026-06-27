"""
    SCP.logtransform([T=Float64,] counts; scale_factor=10_000, kwargs...) -> Job

Apply log transformation: `log(1 + x * scale_factor / total_counts)`. Returns a
`DataMatrix` with the transformed matrix. The element type of the resulting matrix is `T`.

(TODO: Add example.)

See also [`sctransform`](@ref), [`normalize_matrix`](@ref).
"""
function logtransform(T::DataType, counts; scale_factor=10_000, kwargs...)
	create_job(Impl.DataMatrixFunction(Impl.logtransform), T, counts; scale_factor, kwargs...)
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
	create_job(Impl.DataMatrixFunction(Impl.sctransform), T, counts; kwargs...)
end
sctransform(counts; kwargs...) = sctransform(Float64, counts; kwargs...)
