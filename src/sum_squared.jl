col_sum_squared_spec(X) =
	cached(create_spec(SCPCore.col_sum_squared, X; __version=v"0.1.1"))
row_sum_squared_spec(X) =
	cached(create_spec(SCPCore.row_sum_squared, X; __version=v"0.1.1"))


sum_squared_to_var_spec(s2, n) =
	create_spec(SCPCore.sum_squared_to_var, s2, n; __version=v"0.1.0")


function compute_variance(action::Action, X; assume_centered::Bool, col="variance", project=:no)
	@assert project in (:no, :yes)
	# variance/std/relative_std are computed assuming a mean of zero, so the caller must
	# confirm the data is mean-centered. `false` is not supported.
	assume_centered || throw(ArgumentError("assume_centered must be `true`: Consider using `normalize_matrix` to compute variance/std/relative_std."))
	project == :yes && (X = action(X))
	matrix = get_matrix_spec(X)
	s2 = row_sum_squared_spec(matrix)
	n = fetched(nobs_spec(X))
	values = cached(sum_squared_to_var_spec(s2, n))
	table_hcat_spec(id_column_spec(get_var_spec(X)), create_table_spec(col => values))
end

variance_spec(X; kwargs...) = create_spec(Projectable(compute_variance), X; kwargs...)


"""
	Jobs.variance(data; assume_centered, col, project)

Computes the variance of each variable in `data`, and returns a table with IDs and variances.
* `assume_centered` (required) must be set to `true` to confirm that `data` is mean-centered; the variance is computed assuming a mean of zero.
* `col` is the name of the annotation column, defaults to "variance".
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the variance of the base data set, and if `:yes`, it will compute the variance of the projected data set.

See also: [`Jobs.normalize_matrix`](@ref), [`Jobs.std`](@ref), [`Jobs.relative_std`](@ref)
"""
function Jobs.variance(X; kwargs...)
	variance_spec(X; kwargs...)
end


function compute_std(::Preprocessing, X; assume_centered::Bool, col="std", project=:no)
	transform_annotation_spec(sqrt, variance_spec(X; assume_centered, project); new_name=col)
end

std_spec(X; kwargs...) = create_spec(Preprocess(compute_std), X; kwargs...)

"""
	Jobs.std(data; assume_centered, col, project)

Computes the standard deviation of each variable in `data`, and returns a table with IDs and variances.
* `assume_centered` (required) must be set to `true` to confirm that `data` is mean-centered; the std is computed assuming a mean of zero.
* `col` is the name of the annotation column, defaults to "std".
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the std of the base data set, and if `:yes`, it will compute the std of the projected data set.

See also: [`Jobs.normalize_matrix`](@ref), [`Jobs.variance`](@ref), [`Jobs.relative_std`](@ref)
"""
function Jobs.std(X; kwargs...)
	std_spec(X; kwargs...)
end


function compute_relative_std(::Preprocessing, X; assume_centered::Bool, col="relative_std", project=:no)
	std_table = std_spec(X; assume_centered, project)
	max_std = prefetched(apply_spec(maximum, value_column_data_spec(std_table)))
	transform_annotation_spec(Base.Fix2(/, max_std), std_table; new_name=col)
end

relative_std_spec(X; kwargs...) = create_spec(Preprocess(compute_relative_std), X; kwargs...)

"""
	Jobs.relative_std(data; assume_centered, col, project)

Computes the standard deviation of each variable in `data` relative to the maximum standard deviation,
returning a table with IDs and values in [0,1].

Useful for filtering variables: `Jobs.filter_var(Jobs.relative_std(data) => >=(f), data)` keeps
only variables whose std is at least a fraction `f` of the highest-std variable.

* `assume_centered` (required) must be set to `true` to confirm that `data` is mean-centered; the std is computed assuming a mean of zero.
* `col` is the name of the annotation column, defaults to "relative_std".
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the std of the base data set, and if `:yes`, it will compute the std of the projected data set.

See also: [`Jobs.normalize_matrix`](@ref), [`Jobs.variance`](@ref), [`Jobs.std`](@ref)
"""
function Jobs.relative_std(X; kwargs...)
	relative_std_spec(X; kwargs...)
end
