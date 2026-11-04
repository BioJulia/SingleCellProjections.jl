---
name: singlecellprojections-usage
description: Usage instructions for SingleCellProjections.jl (SCP), a Julia package for analysis of single-cell expression data.
---

# SingleCellProjections usage

SingleCellProjections.jl (SCP) analyses single-cell expression data, and can project new
samples onto an existing analysis.
It builds on ReproducibleJobs.jl: SCP functions don't compute anything, they return a `Job`,
a recipe for what to compute.
`fetch!(job)` computes the result.

Results are memoized (in memory, and on disk for some jobs), so we can express analyses as they
should be performed without paying the full computational cost every time.
Computations are lazy.
There is no need to keep results around as variables in Julia, because previous results can be
fetched quickly when needed (keeping them around can waste memory).
Only `fetch!` what is needed: often an end result, or just its annotations (see below).


## Abbreviations

* obs - observations (typically cells)
* var - variables (typically genes)
* ADT - Antibody-Derived Tags (CITE-seq protein measurements)
* DAG - Directed Acyclic Graph
* PCA - Principal Component Analysis
* LOF - Local Outlier Factor


## Packages

At startup, load these packages:
* SingleCellProjections.jl
  (do both `using SingleCellProjections` and `import SingleCellProjections as SCP`)
* ReproducibleJobs.jl (`fetch!`, `print_spec`)
* CSV.jl (needed by `SCP.load_csv`) and DataFrames.jl
* UMAP.jl or TSne.jl, only when computing those embeddings

The functions returning Jobs are public but not exported: call them as `SCP.pca` etc.
A few helpers are exported, e.g. `twogroup_covariate`.

SCP relies heavily on threading, so start Julia with threads (e.g. `julia -t auto`).


## Data matrices

A `DataMatrix` is an annotated matrix with variables as rows and observations as columns.

* `data.var` / `data.obs` - `DataFrame`s with variable / observation annotations.
  The first column is always the ID column, and IDs must be unique.
* `data.matrix` - the matrix. Its type depends on the stage of the analysis: sparse counts,
  a lazy Matrix Expression after transformation/normalization (e.g. `A+B₁B₂B₃`), and a dense
  `Matrix{Float64}` after PCA or embedding. It's rarely needed directly.

A `DataMatrix` Job is split into three component Jobs, so we can work with one part without
computing the others:

* `SCP.get_obs(job)` / `SCP.get_var(job)` - just the annotations.
  Much quicker to fetch than the whole `DataMatrix`.
* `SCP.get_matrix(job)` - just the matrix.
* `SCP.nobs(job)` / `SCP.nvar(job)` - the number of observations / variables.

Fetched results are shared between Jobs: never mutate them.
Use `copy` before changing a fetched `DataFrame`.


## Loading data

`SCP.load_counts` loads 10x `.h5` files (or CellRanger `.mtx[.gz]` files):

```julia
raw_counts = SCP.load_counts(paths; sample_names)
```

* Cell IDs are `"$(sample_name)_$(barcode)"`, and the `sample_name` obs column holds the sample.
* By default only Gene Expression features are kept.
  For ADT, use `prefilter="feature_type"=>isequal("Antibody Capture")`.

To pick and load samples from one or more datasets, see [sample-tables.md](sample-tables.md).

Annotations are tables whose first column holds the IDs.
Load them with `SCP.load_csv(path)` (CSV or TSV) and add them with `SCP.annotate_obs(data, annots)`.


## Standard pipeline

Counts are QC-annotated and filtered, transformed, normalized (centered, regressing out
covariates such as `fraction_mt`), reduced by PCA, and embedded in 2 or 3 dimensions for
visualization:

```julia
# QC annotations
counts = SCP.var_counts_fraction(raw_counts, "fraction_mt", "name"=>startswith("MT-"))
counts = SCP.var_counts_sum(counts, "total_RNA_count")
counts = SCP.var_counts_sum(!iszero, counts, "nonzero_RNA_count")
counts = SCP.obs_counts_sum(!iszero, counts, "nonzero_cell_count")

# Quality filtering
filtered = SCP.filter_obs("fraction_mt" => <(0.15), counts)
filtered = SCP.filter_obs("total_RNA_count" => >(1000), filtered)
filtered = SCP.filter_obs("nonzero_RNA_count" => >(500), filtered)

transformed = SCP.sctransform(filtered)
normalized = SCP.normalize_matrix(transformed, "fraction_mt")
reduced = SCP.pca(normalized; nsv=40)
fl = SCP.force_layout(reduced; ndim=3, seed=4567, k=100, k_projection=25)
```

The parameter values are starting points; suitable values depend on the dataset.

* **Filtering** - `SCP.filter_obs`/`SCP.filter_var` take a `column => predicate` pair, or an
  annotation table with a predicate, e.g.
  `SCP.filter_obs(SCP.annotation(annots, "doublet_score") => <=(0.65), counts)`.
  Cells missing from the table get `missing`, which the predicate must handle
  (e.g. `<=(0.65) ∘ Base.Fix2(coalesce, 0.0)`).
* **Transformation** - `SCP.sctransform` is the default choice.
  `SCP.logtransform` (`log(1 + x * scale_factor / total_counts)`) keeps the matrix sparse, as the
  Mann-Whitney test needs.
* **Normalization** - always normalize before PCA; at the very least the data must be centered.
  More obs annotations (numerical or categorical) can be regressed out.
* **PCA** - `nsv` is the number of components, typically 40-100.
* **Embeddings** - `SCP.force_layout`, `SCP.umap` (needs UMAP.jl) and `SCP.tsne` (needs TSne.jl),
  all with `ndim` 2 or 3.
  Rotate an embedding so that chosen cell groups point along the axes (the first group up):

  ```julia
  transform = SCP.find_optimal_coord_transform(fl, "celltype"=>isequal("HSC"), "celltype"=>isequal("T-cells"), "celltype"=>isequal("B-cells"))
  fl = SCP.transform_coords(fl, transform; keep_var=true)
  ```

Analyses should be performed on the normalized or PCA-reduced data, not on embeddings.
Embeddings are for visualization.


## Projections

`SCP.project` applies an analysis built on base data to new data.
It takes a Job and one or more `old => new` replacements, and repeats every step between them with
the models fitted on the base data (transformation parameters, normalization, PCA, embedding):

```julia
proj_raw_counts = SCP.load_counts(proj_paths; sample_names=proj_names)
proj_fl = SCP.project(fl, raw_counts=>proj_raw_counts)
```

* The result contains only the projected observations; plot the base data as background.
* Variable IDs must match between the base and the new data (same features).
* Any Job built on the base data can be projected, e.g. tests or outlier scores, not only embeddings.
* Give one pair per input that differs, e.g. both RNA and ADT counts.
* Only the steps after the replaced Job are repeated.
  E.g. to skip the quality filtering for the projected cells, replace the filtered Job:

  ```julia
  proj_counts = SCP.project(counts, raw_counts=>proj_raw_counts)
  proj_reduced = SCP.project(reduced, filtered=>proj_counts)
  ```


## Annotation tables

Annotation tables are tables (`DataFrame`s, or Jobs returning them) whose first column is the ID.

* `SCP.annotation(table, col)` - a two-column table with the ID column and `col`.
  Commonly passed to filtering, covariates and tests.
* `SCP.get_columns(table, cols...)` - a table with the given columns.
* `SCP.value_column_data(table)` / `SCP.id_column_data(table)` / `SCP.column_data(table, col)` -
  a column as a vector.
* `SCP.annotate_obs(data, table)` / `SCP.annotate_var(data, table)` - left-join a table onto
  `obs` / `var`.
* `SCP.add_obs_column(data, name, values)` - add a column whose values are in `obs` order.
* `SCP.transform_annotation(f, table)` - apply `f` to the values of a two-column table.
* `SCP.table_leftjoin(a, b)` / `SCP.create_table(col => values, ...)` - join or create tables.

Get an obs annotation as a vector, in the same order as the observations:

```julia
celltype = fetch!(SCP.value_column_data(SCP.annotation(SCP.get_obs(data), "celltype")))
```


## Inspecting

* `fetch!(SCP.get_var(transformed))` - the variables (e.g. gene names) left after transformation.
* `fetch!(SCP.get_obs(data))` - the observation annotations.
* `print_spec(job; [maxdepth])` - print a Job as a DAG (shown as a tree with references).
  Output can be redirected to an IO object.


## Other analyses

* Statistical tests - see [statistical-tests.md](statistical-tests.md).
* `SCP.pseudobulk(normalized, "sample_name")` - aggregate cells into one observation per sample
  (more covariates give one observation per combination).
* `SCP.population_matrix(SCP.get_obs(data), "sample_name"; new_var_covariates="celltype")` -
  the fraction of cells of each cell type, per sample.
* `SCP.local_outlier_factor(reduced, normalized; k=10)` - an LOF score per cell: how much of an
  outlier the cell is compared to its neighbors.
  When projected, only neighbors in the base data are considered, so it shows cells unlike any
  in the base data.
* `SCP.signature(transformed, "name"=>in(genes), "my_signature")` - a signature score per cell,
  the first principal component of the given genes after normalization.
* `SCP.transfer_annotation(reduced, proj_reduced, "celltype"; k=10)` - transfer an obs
  annotation of the base data to the projected cells by kNN voting, with a confidence score.


## Plotting

SCP has no plotting functions; use any plotting library.
Fetch the coordinates and an annotation vector in the same order as the observations:

```julia
coords = fetch!(SCP.get_matrix(fl))  # ndim × nobs Matrix
celltype = fetch!(SCP.value_column_data(SCP.annotation(SCP.get_obs(fl), "celltype")))
```

For projections, plot the base data's coordinates as background.
