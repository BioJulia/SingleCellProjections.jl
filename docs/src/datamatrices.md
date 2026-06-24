```@meta
CurrentModule = SingleCellProjections
ShareDefaultModule = true
```

# Data Matrices

`DataMatrix` objects — annotated matrices where rows are variables and columns are observations — are central in SingleCellProjections.jl.

Let's load a sample to see what a `DataMatrix` looks like:

```@example
using SingleCellProjections
using ReproducibleJobs
using SparseArrays

sample_path = joinpath("samples", "AML28.h5")
```
```@setup
using ..SingleCellDocUtils
sample_path = SingleCellDocUtils.get_lilljebjorn_sample_path("AML28")
```
```@example
counts = Jobs.load_counts(sample_path; sample_names="AML28")
c = fetch!(counts)
```

The display shows the matrix size (number of variables and observations), a brief description of the matrix contents, and an overview of available variable and observation annotations.


## Variables
Variables, or `var` for short, are typically genes or features.
The variables are stored as a [`DataFrame`](https://dataframes.juliadata.org/stable/) and can be accessed by:
```@example
c.var[1:6, :]
```


## Observations
Observations, or `obs` for short, are typically cells.
The observations are stored as a [`DataFrame`](https://dataframes.juliadata.org/stable/) and can be accessed by:
```@example
c.obs[1:6, :]
```


## IDs
Each variable and each observation must have a unique ID.
The first column of the `var` and `obs` DataFrames is always the ID column.

Most of the time, IDs are handled automatically by SingleCellProjections.jl.
Sometimes, you need to make sure IDs are unique when loading or merging data matrices.
In particular, when loading a `DataMatrix` that should be projected onto another `DataMatrix`, the user must ensure that variable IDs match.


## Matrix
The matrix can be accessed by `data.matrix`.
Depending on the stage of analysis, different kinds of matrices (or matrix-like objects) are used.
Most of this complexity is hidden from the user, but internally SingleCellProjections.jl depends on this functionality to be fast and to reduce memory usage.

!!! warning "Read-only"
    SingleCellProjections.jl will reuse matrices when possible, in order to reduce memory usage.
    E.g. [`Jobs.normalize_matrix`](@ref) will reuse and extend the Matrix Expression of the source `DataMatrix`, without creating a copy of the actual data.
    When matrices are reused/copied is considered an implementation detail, and can change at any time.
    Users of SingleCellProjections.jl should thus consider the matrices to be "read-only".
    This should rarely present problems in practice.

Roughly, the matrix types used at different stages are:

1. Counts — [`SparseMatrixCSC`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/) (often in a blocked format)
2. Transformed and normalized data — [Matrix Expressions](@ref)
3. PCA result — `Matrix{Float64}`
4. ForceLayout/UMAP/t-SNE result — `Matrix{Float64}`


## Jobs and DataMatrices

A `DataMatrix` `Job` is internally split into three component Jobs: one for the matrix, one for `var`, and one for `obs`. You can access these with:
- `Jobs.get_matrix(job)` — the matrix component
- `Jobs.get_var(job)` — the variable annotations
- `Jobs.get_obs(job)` — the observation annotations

Operations that only affect one component leave the others unchanged. For example, `Jobs.logtransform` only transforms the matrix — it passes `var` and `obs` through without modification. This means the var and obs Jobs are literally the same object (identical by `===`) before and after the transformation:

```@example
transformed = Jobs.logtransform(counts)

# var is unchanged by logtransform — same Job object
forward!(Jobs.get_var(transformed)) === forward!(Jobs.get_var(counts))
```

```@example
# matrix is different — logtransform created a new matrix Job
forward!(Jobs.get_matrix(transformed)) === forward!(Jobs.get_matrix(counts))
```

This splitting is what makes the caching and projection system efficient — unchanged components are never recomputed or stored redundantly.

!!! warning "Do not mutate results"
    Since components are shared across Jobs, mutating a fetched result (e.g., modifying a column in `data.obs`) would corrupt other Jobs that reference the same underlying object. Always treat results from `fetch!` as read-only.
