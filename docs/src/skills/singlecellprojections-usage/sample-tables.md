# Sample tables

A sample table describes the samples of one or more datasets, so that we can pick samples and
load them.
No special package or type is needed: the flow below uses CSV.jl, DataFrames.jl and
`SCP.load_counts`.
Every step is optional and ordinary Julia; skip or replace any step that doesn't fit.


## The contract

A sample table is a `DataFrame` whose **first column is the sample ID**.
That is the only requirement.

* Column names are free. The ID is found by position, as in SingleCellProjections tables.
* Other columns are whatever the project finds useful (e.g. `condition`, `tissue`, `path`).
  Not every dataset needs every column.
* `missing` values are allowed, e.g. the path of a sample that isn't downloaded.
* The sample ID passed to `load_counts` becomes the cell-ID prefix:
  cell IDs are `"$(sample_name)_$(barcode)"`.
  Loading errors on duplicate cell IDs, so datasets loaded together need distinct sample IDs.
  An ID may repeat within a table when one sample is split across files.


## The flow

1. **Load a dataset's table from disk.**
   This is the portable part: one table per dataset, with sample-level annotations and the
   dataset's own (local) sample IDs. No machine-specific paths.
2. **Add file paths.** Build them from the local IDs and a machine-specific folder.
3. **Make global IDs (optional).** Alter the IDs if needed, so that they are unique across the
   datasets you combine. Do this after building paths, since file names use the local IDs.
4. **Combine datasets (optional).** `vcat` the tables.
5. **Select samples** with `filter` or `subset`.
6. **Load** the selected samples with `SCP.load_counts`.

Here our own dataset is combined with an external one:

```julia
using CSV, DataFrames
import SingleCellProjections as SCP

# 1. Portable tables with local IDs
ours = CSV.read("ours.tsv", DataFrame)
external = CSV.read("external.tsv", DataFrame)

# 2. Paths from local IDs
ours.path = joinpath.(ours_folder, ours.sample_name .* ".h5")
external.path = joinpath.(external_folder, external.sample_name .* ".h5")

# 3. Global IDs
external.sample_name .*= "_external"

# 4. Combine: columns a dataset lacks become `missing`
st = vcat(ours, external; cols=:union, source=:dataset => ["ours", "external"])

# 5. Select
sel = filter(:condition => isequal("control"), st)

# 6. Load
counts = SCP.load_counts(sel.path; sample_names=sel.sample_name)
```

A single dataset needs only steps 1, 2, 5 and 6.
To try a single new sample, no table is needed at all:
`SCP.load_counts("SampleA.h5"; sample_names="SampleA")`.


## Paths

How the folder is found is up to the project, e.g. a variable, an environment variable or a
[Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl) setting.
If the folder isn't known, use a `missing` path so that the table can still be used for
selection:

```julia
df.path = isnothing(folder) ? missings(String, nrow(df)) : joinpath.(folder, df.sample_name .* ".h5")
```

`load_counts` doesn't accept `missing` paths, so check before loading if needed:

```julia
any(ismissing, sel.path) && error("Paths missing for: ", join(sel.sample_name[ismissing.(sel.path)], ", "))
```

If files are named differently, build the path column any way that works, e.g. from another
column, or by listing a folder.


## Selecting samples

Combined tables contain many `missing` values.
Use predicates that return `true`/`false` for `missing`, or `subset` with `skipmissing=true`:

| Works with `missing` | Errors with `missing` |
| --- | --- |
| `filter(:condition => isequal("control"), st)` | `filter(:condition => ==("control"), st)` |
| `filter(:condition => in(Set(["treated_1h", "treated_24h"])), st)` | `filter(:condition => in(["treated_1h", "treated_24h"]), st)` |
| `subset(st, :condition => ByRow(==("control")); skipmissing=true)` | `subset(st, :condition => ByRow(contains(r"^treated")); skipmissing=true)` |
| `subset(st, :condition => ByRow(passmissing(contains(r"^treated"))); skipmissing=true)` | |

Combine conditions with several pairs in `subset`, or with a function of several columns in
`filter`:

```julia
sel = subset(st, :tissue => ByRow(isequal("blood")), :dataset => ByRow(!=("external")))
sel = filter([:tissue, :dataset] => (t, d) -> isequal(t, "blood") && d != "external", st)
```

The IDs of the selection are its first column, `sel[!, 1]`.


## Sample annotations

Since the first column is the ID, a sample table works directly as an annotation table for
sample-level data such as pseudobulk, e.g. `SCP.annotation(st, "condition")`.
