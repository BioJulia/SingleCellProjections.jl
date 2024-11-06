# SingleCellProjections.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.3] - 2024-11-06

### Added

* Package extension for `Muon.jl` that allows loading data from .h5ad files using the functions `create_datamatrix`, `create_var` and `create_obs`.

### Fixed

* Deprecated old `loadh5ad` function that only supported some versions of the .h5ad format.

## [0.4.2] - 2024-09-27

### Fixed

* Fix bug when merging samples with different sets of features, which sometimes could cause unsorted row values and thus sparse matrices with invalid internal representation.
* `project` now handles `kwargs` that are only needed for some of the projection steps (by simply ignoring them when not needed).

## [0.4.1] - 2024-08-27

### Added

* `local_outlier_factor`, `local_outlier_factor_table`, `local_outlier_factor_projection`, `local_outlier_factor_projection_table`

### Fixed

* Updated tutorial to reflect changes in SingleCellProjections v0.4.

## [0.4.0] - 2024-06-20

### Breaking

* DataMatrix will now always use the first column of var/obs annotations as ID. (Multiple ID columns are no longer supported.)
* `load_counts` - The default obs ID column name is now "cell_id" (was "id" before).
* `load10x` - default to using only first column (id) as unique identifier. Specify e.g. `var_id="var_id"=>["id", "feature_type"]` to merge multiple columns to create the ID.
* `load10x` - default to using first column (barcode) as unique identifier.
* `load10x` - no longer supports `copy_obs_col` kwarg.
* `set_var_id_cols!` is replaced with `set_var_id_col!` (since there is only one ID column).
* `set_obs_id_cols!` is replaced with `set_obs_id_col!` (since there is only one ID column).
* Update to SCTransform 0.2, which handles `logcellcounts` better when there are multiple modalities (e.g. RNA and antibody counts) present in the data.

### Added

* `var_counts_fraction` - Just like `var_counts_fraction!`, but not modifying the object in place.
* `var_counts_sum` and `var_counts_sum!` - For summing over selected variables. Useful for counting e.g. total RNA expression and finding number of expressed features.
* Added support for using external annotations where applicable (filter, transforms, normalization, statistical tests, var_counts_fraction!, var_counts_sum!)
* Added experimental (thus yet unexported) `Annotations` struct, that wraps a `DataFrame` with IDs in the first column, and ensures that ID remain when accessing columns. (So that the resulting object can be leftjoined to `data.obs`/`data.var`.)

### Fixed

* Add compat for weakdeps (UMAP, TSne, PrincipalMomentAnalysis).
* SVDModel now only stores `U` and `S` since `V` is not needed for projection.

## [0.3.9] - 2024-03-04

### Fixed

* Relax `===` to `==` when comparing some models. (This fixes a bug occurring when a model is saved to disk using e.g. JLD2 and the loaded again.)

## [0.3.8] - 2024-02-22

### Added

* `svd`, `force_layout` and `pma` now supports `seed` kwarg. To use it, `StableRNGs` must be loaded.

## [0.3.7] - 2023-12-19

### Fixed

* `load_counts` - You can now pass a single filename to load a single file (previously arrays were required, but the error message was confusing).

## [0.3.6] - 2023-12-15

### Added

* `local_outlier_factor!` - Compute the Local Outlier Factor (LOF) for each observation in a DataMatrix. Supports finding neighbors in a low dimensional space (e.g. after PCA or UMAP), but computing distances in a high dimensional space (e.g. after normalization).
* `local_outlier_factor_projection!` - Compute the Local Outlier Factor (LOF) for each observation in a DataMatrix. Only points in the `base` data set are considered as neighbors.

### Fixed

* `knn_adjacency_matrix` - kwarg `make_symmetric` must now be specified by the caller.


## [0.3.5] - 2023-12-12

### Added

* `pseudobulk`: function used to collapse a DataMatrix into a smaller DataMatrix by averaging over groups of observations.

### Fixed

* Add stdlib compat


## [0.3.4] - 2023-09-13

### Fixed

* Add compat with HDF5.jl v0.17

## [0.3.3] - 2023-08-16

### Fixed

* UMAP, TSne and PrincipalMomentAnalysis support now uses Package Extensions (on Julia 1.9+)
* Compat bump for SingleCell10x which should reduce loading time and memory usage when reading from .h5 files

## [0.3.2] - 2023-07-17

### Fixed

* Bug fix: Add missing method for `SCTransformModel`.

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
