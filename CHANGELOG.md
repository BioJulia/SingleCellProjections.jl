# SingleCellProjections.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

* `normalize_matrix`: Categorical coviariates with `missing` values will now error.
* `differentialexpression`: Removed function. Differential expression is now done with `ftest`, `ttest` or `mannwhitney` instead.
* `logtransform` and `tf_idf_transform` now defaults to only keeping features with `feature_type` "Gene Expression" (if `feature_type` is present as a variable annotation).
