# Breaking changes in SingleCellProjections.jl

SingleCellProjections.jl, [like other Julia packages](https://pkgdocs.julialang.org/v1/compatibility/) uses [semantic versioning](https://semver.org) (semver).

Whenever there is a breaking release, the breaking changes will be listed in this file.

## 0.3

* `normalize_matrix`: Categorical coviariates with `missing` values will now error.
