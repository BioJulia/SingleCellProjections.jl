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
	create_job(Preprocess(Impl.ftest), data, h1; kwargs...)
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
	create_job(Preprocess(Impl.ttest), data, h1; kwargs...)
end


"""
    SCP.mannwhitney(data, column, [group_a, group_b]; h1_missing=:skip, kwargs...) -> Job

Perform a Mann-Whitney U-test (a.k.a. Wilcoxon rank-sum test) between two groups of
observations, for each variable. The U statistic is corrected for ties, and p-values are
computed using a normal approximation. Returns a table with variable IDs, U statistics and
p-values, sorted by significance (see below).

`data` must contain a sparse matrix. It is recommended to first [`logtransform`](@ref) (or
`tf_idf_transform`) the raw counts.

`column` selects a column in `data.obs` that determines group membership:
* If neither `group_a` nor `group_b` is given, `column` must have exactly two unique values (ignoring `missing`).
* If only `group_a` is given, observations equal to `group_a` are compared against all others (ignoring `missing`).
* If both are given, observations equal to `group_a` are compared against those equal to `group_b`.

Keyword arguments:
* `h1_missing=:skip` - `:skip` excludes `missing` values in `column`; `:error` throws if any are present.
* `statistic_col="U"` / `pvalue_col="pValue"` / `z_col=nothing` - output column names (set to `nothing` to omit; `z` is omitted by default).
* `do_sort=true` - sort variables by `|z|` (most significant first).

Results are sorted by the absolute standardized statistic `|z|`, where `z = (U - n1*n2/2)/σ`. This
orders variables by significance without the underflow that sorting by `pValue` suffers (p-values
collapse to `0` for strongly-separated variables). The signed `z` (available via `z_col`) is monotone
with the p-value and also indicates the direction of the effect.

The test is projectable: when projecting onto other data, the group labels resolved here are
reused and the test is recomputed on the projected observations.

See also [`ftest`](@ref), [`ttest`](@ref), [`logtransform`](@ref).
"""
function mannwhitney(data, column, args...; kwargs...)
	create_job(Preprocess(Impl.mannwhitney), data, column, args...; kwargs...)
end
