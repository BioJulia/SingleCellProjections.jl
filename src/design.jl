"""
    categorical_covariate()

Create a categorical covariate description. Used in `Pair`s to specify that a column
should be treated as categorical, e.g. `"celltype" => categorical_covariate()`.

This is the default for string columns, so explicit use is rarely needed.

See also [`numerical_covariate`](@ref), [`twogroup_covariate`](@ref).
"""
categorical_covariate() = SCPCore.CategoricalCovariateDesc()

"""
    numerical_covariate()

Create a numerical covariate description. Used in `Pair`s to specify that a column
should be treated as numerical, e.g. `"age" => numerical_covariate()`.

This is the default for numeric columns, so explicit use is rarely needed.

See also [`categorical_covariate`](@ref), [`twogroup_covariate`](@ref).
"""
numerical_covariate() = SCPCore.NumericalCovariateDesc()

"""
    twogroup_covariate(group_a, group_b=nothing)

Create a two-group covariate description for comparing two specific groups within a
categorical column. `group_a` and `group_b` specify the two group values to compare.
If `group_b` is `nothing`, all observations not in `group_a` are treated as the other group.

See also [`categorical_covariate`](@ref), [`numerical_covariate`](@ref).
"""
twogroup_covariate(group_a, group_b=nothing) = SCPCore.TwoGroupCovariateDesc(group_a, group_b)


"""
    SCP.designmatrix(data, covariates...; center=true, kwargs...) -> Job

Construct a design matrix from observation covariates. Covariates can be column names
(strings) or `Pair`s of column name and covariate description. Used internally by
`SCP.normalize_matrix`.

See also [`normalize_matrix`](@ref), [`negative_regression_matrix`](@ref).
"""
function designmatrix(data, args...; kwargs...)
	Impl.designmatrix_job(data, args...; kwargs...)
end
