col_sum_squared_spec(X) =
	cached(create_spec(SCPCore.col_sum_squared, X; __version=v"0.1.1"))
row_sum_squared_spec(X) =
	cached(create_spec(SCPCore.row_sum_squared, X; __version=v"0.1.1"))


sum_squared_to_var_spec(s2, n) =
	create_spec(SCPCore.sum_squared_to_var, s2, n; __version=v"0.1.0")


function compute_variance(action::Action, X; col="variance", project=:no)
	@assert project in (:no, :yes)
	if project == :yes
		X = action(X)
	end
	matrix = get_matrix_spec(X)
	s2 = row_sum_squared_spec(matrix)
	n = fetched(nobs_spec(X))
	values = cached(sum_squared_to_var_spec(s2, n))
	table_hcat_spec(id_column_spec(get_var_spec(X)),
	                create_table_spec(col => values))
end

variance_spec(X; kwargs...) = create_spec(Projectable(compute_variance), X; kwargs...)


"""
	Jobs.variance(data; col, project)

Computes the variance of each variable in `data`, and returns a table with IDs and variances.
* `col` is the name of the annotation column, defaults to "variance".
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the variance of the base data set, and if `:yes`, it will compute the variance of the projected data set.

!!! note
	`data` must be mean-centered. E.g. by using `normalize_matrix` before calling `Jobs.variance`.

See also: [`Jobs.std`](@ref)
"""
function Jobs.variance(X; kwargs...)
	variance_spec(X; kwargs...)
end


compute_std(::Preprocessing, X; col="std", project=:no) =
	transform_annotation_spec(sqrt, variance_spec(X; project); new_name=col)

std_spec(X; kwargs...) = create_spec(Preprocess(compute_std), X; kwargs...)

"""
	Jobs.std(data; col, project)

Computes the standard deviation of each variable in `data`, and returns a table with IDs and variances.
* `col` is the name of the annotation column, defaults to "std".
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the std of the base data set, and if `:yes`, it will compute the std of the projected data set.

!!! note
	`data` must be mean-centered. E.g. by using `normalize_matrix` before calling `Jobs.std`.

See also: [`Jobs.variance`](@ref)
"""
function Jobs.std(X; kwargs...)
	std_spec(X; kwargs...)
end


function compute_relative_std(::Preprocessing, X; col="relative_std", project=:no)
	s = std_spec(X; project)
	values = value_column_data_spec(s)
	max_std = prefetched(apply_spec(maximum, values))
	transform_annotation_spec(Base.Fix2(/, max_std), s; new_name=col)

	# DEBUG - this doesn't trigger the problem
	# transform_annotation_spec(sqrt, prefetched(s); new_name=col)
end

relative_std_spec(X; kwargs...) = create_spec(Preprocess(compute_relative_std), X; kwargs...)

"""
	Jobs.relative_std(data; col, project)

Computes the standard deviation of each variable in `data` relative to the maximum standard deviation,
returning a table with IDs and values in [0,1].

Useful for filtering variables: `Jobs.filter_var(Jobs.relative_std(data) => >=(f), data)` keeps
only variables whose std is at least a fraction `f` of the highest-std variable.

* `col` is the name of the annotation column, defaults to "relative_std".
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the std of the base data set, and if `:yes`, it will compute the std of the projected data set.

!!! note
	`data` must be mean-centered. E.g. by using `normalize_matrix` before calling `Jobs.relative_std`.

See also: [`Jobs.std`](@ref)
"""
function Jobs.relative_std(X; kwargs...)
	relative_std_spec(X; kwargs...)
end
