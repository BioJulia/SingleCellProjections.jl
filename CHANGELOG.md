# SingleCellProjections.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2023-07-17

### Added

* Float32 support: `sctransform`, `logtransform` and `tf_idf_transform` now supports an optional type argument `T` which controls the eltype of the sparse transformed matrix. By setting it to `Float32` it is possible to reduce memory usage with little impact on results, since downstream computations are still performed in `Float64` precision.

## [0.3] - 2023-06-23

### Breaking

* `normalize_matrix`: Categorical coviariates with `missing` values will now error.
* `differentialexpression`: Removed function. Differential expression is now done with `ftest`, `ttest` or `mannwhitney` instead.
* `logtransform` and `tf_idf_transform` now defaults to only keeping features with `feature_type` "Gene Expression" (if `feature_type` is present as a variable annotation).

### Added

* Statistical tests: F-test (ANOVA, Quadratic Regression, etc.), t-tests (Two-Group comparison, linear regression etc.) and MannWhitney U-test (Wilcoxon rank-sum-test).
* Support for TwoGroup covariates (also useful for `normalize_matrix`).
