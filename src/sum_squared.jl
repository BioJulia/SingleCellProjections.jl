"""
    SCP.variance(data; assume_centered, col="variance", project=:no) -> Job

Compute the variance of each variable in `data`, returning a table with IDs and variances.
* `assume_centered` (required) must be set to `true` to confirm that `data` is mean-centered; the variance is computed assuming a mean of zero.
* `col` is the name of the annotation column, defaults to `"variance"`.
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the variance of the base data set, and if `:yes`, it will compute the variance of the projected data set.

See also [`std`](@ref), [`relative_std`](@ref), [`normalize_matrix`](@ref).
"""
function variance(X; kwargs...)
	Impl.variance_job(X; kwargs...)
end


"""
    SCP.std(data; assume_centered, col="std", project=:no) -> Job

Compute the standard deviation of each variable in `data`, returning a table with IDs and values.
* `assume_centered` (required) must be set to `true` to confirm that `data` is mean-centered; the std is computed assuming a mean of zero.
* `col` is the name of the annotation column, defaults to `"std"`.
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the std of the base data set, and if `:yes`, it will compute the std of the projected data set.

See also [`variance`](@ref), [`relative_std`](@ref), [`normalize_matrix`](@ref).
"""
function std(X; kwargs...)
	Impl.std_job(X; kwargs...)
end


"""
    SCP.relative_std(data; assume_centered, col="relative_std", project=:no) -> Job

Compute the standard deviation of each variable in `data` relative to the maximum standard deviation,
returning a table with IDs and values in [0,1].

Useful for filtering variables: `SCP.filter_var(SCP.relative_std(data) => >=(f), data)` keeps
only variables whose std is at least a fraction `f` of the highest-std variable.

* `assume_centered` (required) must be set to `true` to confirm that `data` is mean-centered; the std is computed assuming a mean of zero.
* `col` is the name of the annotation column, defaults to `"relative_std"`.
* `project` can be `:no` (default) or `:yes`. If `:no`, it will compute the std of the base data set, and if `:yes`, it will compute the std of the projected data set.

See also [`variance`](@ref), [`std`](@ref), [`normalize_matrix`](@ref).
"""
function relative_std(X; kwargs...)
	Impl.relative_std_job(X; kwargs...)
end
