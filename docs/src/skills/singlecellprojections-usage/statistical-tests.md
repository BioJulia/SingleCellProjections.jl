# Statistical tests

SCP tests each variable (e.g. gene) for differences between groups of observations, and
returns a table with one row per variable, most significant first.

| Test | Use for | Data |
| --- | --- | --- |
| `SCP.mannwhitney` | Two groups (Wilcoxon rank-sum test) | Sparse: `SCP.logtransform`ed counts |
| `SCP.ftest` | Several groups (one-way ANOVA), or comparing models | Transformed, not normalized: `SCP.logtransform`ed or `SCP.sctransform`ed counts |
| `SCP.ttest` | Two groups, or a numerical covariate | Transformed, not normalized: `SCP.logtransform`ed or `SCP.sctransform`ed counts |


## Workflow

1. Decide which groups of cells to compare.
2. Create an annotation table with two columns: the cell ID and the group.
   Give the group column an informative name.
3. Load counts that include all cells to compare, and apply the usual quality filtering.
4. Add the annotation with `SCP.annotate_obs`.
5. `SCP.logtransform` the counts.
6. Do a `SCP.mannwhitney` test to get a table with one row per variable.

```julia
groups = DataFrame("cell_id" => cell_ids, "stage" => stages)  # e.g. "early"/"late"
data = SCP.annotate_obs(filtered, groups)
transformed = SCP.logtransform(data)
result = fetch!(SCP.mannwhitney(transformed, "stage", "early", "late"))
```

Change to an F-test if there are multiple groups, or to a t-test for a numerical covariate.
They take the same log-transformed data, or sctransformed data.


## Mann-Whitney U-test

`SCP.mannwhitney(data, column, [group_a, group_b])` compares two groups of observations, as
given by the obs column `column`:

* No groups given - `column` must have exactly two values.
* Only `group_a` - `group_a` against all other observations.
* Both - `group_a` against `group_b`.

Observations with `missing` in `column` are skipped.
`data` must contain a sparse matrix, so use `SCP.logtransform` rather than `SCP.sctransform`.

The result has the variable IDs, the `U` statistic and `pValue`, sorted by significance.
Add `z_col="z"` to get the signed standardized statistic, which shows the direction of the effect.


## F-test and t-test

```julia
transformed = SCP.logtransform(data)  # or SCP.sctransform(data)
SCP.ftest(transformed, "celltype")                                     # one-way ANOVA
SCP.ttest(transformed, "stage" => twogroup_covariate("early", "late")) # two groups
SCP.ttest(transformed, "age")                                          # numerical covariate
```

* `h1` is the tested covariate, either an obs column name or `column => covariate description`.
  The type (categorical or numerical) is detected from the column, or given by
  `categorical_covariate()`, `numerical_covariate()` or `twogroup_covariate(group_a, [group_b])`.
* `h0` gives covariates to control for, e.g. `h0="fraction_mt"`.
  The null model also includes an intercept (`center=true`).
* The t-test result includes a `difference` column, the estimated effect.

Don't `SCP.normalize_matrix` the data first; give covariates in `h0` instead.
Regressing covariates out beforehand removes them without the test knowing, so the degrees of
freedom are wrong and the p-values too small.


## Options for all tests

* `var_cols="name"` - include more `var` columns (e.g. gene names) in the result.
* `statistic_col` / `pvalue_col` - rename the output columns (`nothing` omits them).
* `do_sort=false` - keep the variables in their original order.

The p-values are not adjusted for multiple testing.
