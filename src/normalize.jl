"""
    SCP.negative_regression_matrix(data, design_matrix; kwargs...) -> Job

Compute the negative regression coefficient matrix for normalization. Used internally
by `SCP.normalize_matrix`.

See also [`normalize_matrix`](@ref), [`designmatrix`](@ref).
"""
function negative_regression_matrix(args...; kwargs...)
	Impl.negative_regression_matrix_job(args...; kwargs...)
end


"""
    SCP.normalize_matrix(data, covariates...; center=true, kwargs...) -> Job

Normalize `data` by centering and regressing out covariates. Covariates can be column names
(strings) or `Pair`s of column name and covariate description.

Optional keyword arguments for annotating per-variable statistics:
- `annotate_variance`: Set to true to add a column with per-variable variance.
- `annotate_std`: Set to true to add a column with per-variable standard deviation.
- `annotate_relative_std`: Set to true to add a column with per-variable relative standard deviation.
- `variance_col`: Custom name for the variance column.
- `std_col`: Custom name for the std column.
- `relative_std_col`: Custom name for the relative standard deviation.

# Examples

Center transformed data:
```julia
julia> SCP.normalize_matrix(transformed)
```

Center transformed data and regress out the `fraction_mt` covariate.
```julia
julia> SCP.normalize_matrix(transformed, "fraction_mt")
```

Annotate by relative std:
```julia
julia> SCP.normalize_matrix(transformed; annotate_relative_std=true)
```

Annotate by variance, using a custom name:
```julia
julia> SCP.normalize_matrix(transformed; variance_col="my_variance_column")
```


See also [`sctransform`](@ref), [`logtransform`](@ref), [`designmatrix`](@ref).
"""
function normalize_matrix(data, args...; kwargs...)
	create_job(Preprocess(Impl.normalize_matrix), data, args...; kwargs...)
end
