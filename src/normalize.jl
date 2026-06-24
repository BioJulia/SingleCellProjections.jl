negative_regression_matrix_impl(::Action, data, dm; kwargs...) =
	cached(create_job(SCPCore.negative_regression_matrix, data, dm; kwargs..., __version=v"0.1.0")) # NB: No action, always use original
negative_regression_matrix_impl_job(data, dm; kwargs...) =
	create_job(Projectable(negative_regression_matrix_impl), data, dm; kwargs...)


function negative_regression_matrix(::Mat, data, dm; kwargs...)
	# TODO: check that data and dm IDs match
	negative_regression_matrix_impl_job(get_matrix_job(data), get_matrix_job(dm); kwargs...)
end
negative_regression_matrix(::Var, data, dm; kwargs...) = get_job(Var(), data)
negative_regression_matrix(::Obs, data, dm; kwargs...) = get_job(Obs(), dm)


function negative_regression_matrix_job(data, dm; rtol=nothing)
	rtol = @something rtol sqrt(eps())
	create_job(DataMatrixFunction(negative_regression_matrix), data, dm; rtol)
end
"""
    Jobs.negative_regression_matrix(data, design_matrix; kwargs...) -> Job

Compute the negative regression coefficient matrix for normalization. Used internally
by `Jobs.normalize_matrix`.

See also [`Jobs.normalize_matrix`](@ref), [`Jobs.designmatrix`](@ref).
"""
function Jobs.negative_regression_matrix(args...; kwargs...)
	negative_regression_matrix_job(args...; kwargs...)
end






function normalize_matrix(::Preprocessing, data, args...; center=true,
		variance_col = nothing,
		std_col = nothing,
		relative_std_col = nothing,
		annotate_variance = variance_col !== nothing,
		annotate_std = std_col !== nothing,
		annotate_relative_std = relative_std_col !== nothing,
		kwargs...)
	dm = designmatrix_job(data, args...; center)
	negβT = negative_regression_matrix_job(data, dm; kwargs...)
	dmT = adjoint_job(dm)
	normalized = matrix_sum_job(:A=>data, matrix_product_job(Symbol("(-β)")=>negβT, :X=>dmT))

	if annotate_variance || annotate_std || annotate_relative_std
		center || throw(ArgumentError("Annotating variance/std/relative_std requires center=true (data must be mean-centered);"))
		base = normalized

		if annotate_variance
			variance_col = @something variance_col "variance"
			normalized = Jobs.annotate_var(normalized, Jobs.variance(base; assume_centered=true, col=variance_col))
		end
		if annotate_std
			std_col = @something std_col "std"
			normalized = Jobs.annotate_var(normalized, Jobs.std(base; assume_centered=true, col=std_col))
		end
		if annotate_relative_std
			relative_std_col = @something relative_std_col "relative_std"
			normalized = Jobs.annotate_var(normalized, Jobs.relative_std(base; assume_centered=true, col=relative_std_col))
		end
	end
	normalized
end


"""
    Jobs.normalize_matrix(data, covariates...; center=true, kwargs...) -> Job

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
julia> Jobs.normalize_matrix(transformed)
```

Center transformed data and regress out the `fraction_mt` covariate.
```julia
julia> Jobs.normalize_matrix(transformed, "fraction_mt")
```

Annotate by relative std:
```julia
julia> Jobs.normalize_matrix(transformed; annotate_relative_std=true)
```

Annotate by variance, using a custom name:
```julia
julia> Jobs.normalize_matrix(transformed; variance_col="my_variance_column")
```


See also [`Jobs.sctransform`](@ref), [`Jobs.logtransform`](@ref), [`Jobs.designmatrix`](@ref).
"""
function Jobs.normalize_matrix(data, args...; kwargs...)
	create_job(Preprocess(normalize_matrix), data, args...; kwargs...)
end
