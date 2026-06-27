"""
    SCP.ftest(data, h1; h0=(), center=true, kwargs...) -> Job

Perform an F-test for each variable comparing the full model `h1` against the null
model `h0`. Returns a table with test statistics and p-values.

`h1` and `h0` are covariates specified as column name strings or `Pair`s of column name and
covariate description. The covariate type (categorical/numerical) is normally autodetected. With
a single categorical covariate, this is equivalent to a one-way ANOVA.

(TODO: Examples.)

See also [`ttest`](@ref), [`normalize_matrix`](@ref).
"""
function ftest(data, h1; kwargs...)
	Impl.ftest_job(data, h1; kwargs...)
end


"""
    SCP.ttest(data, h1; h0=(), center=true, kwargs...) -> Job

Perform a t-test for each variable testing the effect of `h1` while controlling for `h0`.
Returns a table with test statistics and p-values. `h1` must be a numerical covariate or a
two-group covariate.

(TODO: Examples.)

See also [`ftest`](@ref), [`normalize_matrix`](@ref), [`twogroup_covariate`](@ref).
"""
function ttest(data, h1; kwargs...)
	Impl.ttest_job(data, h1; kwargs...)
end
