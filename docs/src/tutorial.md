# Tutorial

```@meta
ShareDefaultModule = true # All unnamed setup/example/repl blocks share the same module, so we can just keep adding stuff and reuse old variables
```

The goals of this tutorial are to:

1. Show a standard workflow for analyzing single cell RNA-seq data.
2. Provide a brief overview of how `SingleCellProjections.jl` and `ReproducibleJobs.jl` work.
3. Show how to easily project one data set onto another.


We will use an [Acute Myeloid Leukemia (AML)](https://en.wikipedia.org/wiki/Acute_myeloid_leukemia) data set from the paper:
> Henrik Lilljebjörn, Pablo Peña-Martínez, Hanna Thorsson, Rasmus Henningsson, Marianne Rissler, Niklas Landberg, Noelia Puente-Moncada, Sofia von Palffy, Vendela Rissler, Petr Stanek, Jonathan Desponds, Xiangfu Zhong, Gunnar Juliusson, Vladimir Lazarevic, Sören Lehmann, Magnus Fontes, Helena Ågerstam, Carl Sandén, Christina Orsmark-Pietras, Thoas Fioretos. "[The AML cellular state space unveils *NPM1* immune evasion subtypes with distinct clinical outcomes](https://doi.org/10.1038/s41467-025-66546-6)". Nat Commun 16, 10592 (2025).

You can download the data here:

* [Single Cell RNA-seq Data](https://doi.org/10.17044/scilifelab.23715648) - for this tutorial, we will use some of the .h5 files.
* [Cell annotations](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/Lilljebjorn2025/scRNA_AML.tar.gz)


The data set contains 38 patient samples and 8 samples from healthy donors. The samples contain ~6000 cells on average, with measurements from 32738 genes.




First, we will load [SingleCellProjections.jl](https://BioJulia.github.io/SingleCellProjections.jl) and some other useful packages:

* [ReproducibleJobs.jl](https://github.com/rasmushenningsson/ReproducibleJobs.jl) provides a framework used by `SingleCellProjections` to handle computations, caching etc.
* [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) and [CSV.jl](https://github.com/JuliaData/CSV.jl) we use for handling and loading tabular data.
* [SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl) is used under the hood by SingleCellProjections. We just load it here so we can inspect some of the raw data.

```@example
using SingleCellProjections
using ReproducibleJobs
using CSV
using DataFrames
using SparseArrays
```

```@setup
using ..SingleCellDocUtils


# We might want to enable this later, in particular if we use the same Jobs from multiple documentation pages.
# Empty the scheduler/deduplicator to avoid getting already cached results showing up in outputs when running multiple times (typically when creating docs locally in the REPL).
# empty!(ReproducibleJobs.get_scheduler());
# empty!(ReproducibleJobs.get_scheduler().deduplicator);



# Plotting
using Bonito
using WGLMakie
WGLMakie.activate!()
Makie.inline!(true)   # inline into the page instead of opening a browser
Page()
```

## Loading Data

Let's start by loading two healthy Normal Bone Marrow (NBM) samples. These samples are from the same donor, but one has been enriched for immature (CD34 positive) cells.
This was done because AML patients have more immature cells in their bone marrow than a healthy person.
Combined, the NBM samples make a good reference when looking at AML samples later.

You'll need to change the paths to wherever you have downloaded your files.

```@example
nbm_names = ["NBM10-CD34", "NBM10-MNC"]
nbm_paths = joinpath.("samples", string.(nbm_names, ".h5"))
```

```@setup
nbm_paths = SingleCellDocUtils.get_lilljebjorn_sample_path.(nbm_names)
```


Loading is done using `Jobs.load_counts`.
```@example
raw_counts = Jobs.load_counts(nbm_paths; sample_names=nbm_names)
```
This creates a `Job`, which is a kind of specification or recipe for what to compute. `Job`s are the cornerstone of `ReproducibleJobs.jl` since it makes it possible to reason about computations without performing them. Importantly, some results are cached, so when we come back another day, we do not need to recompute everything again. This mostly happens under the hood, and it will not be the focus of this tutorial.

To actually retrieve the data, we need to use `fetch!`.

!!! warning "fetch!"
    Calling `fetch!` forces a computation (or loading from the cache), and it should thus only be used when needed. Often, the `ReproducibleJobs.jl` machinery can avoid computing/loading results from intermediate steps altogether, so if you only use an end result, only call `fetch!` on that job.

```@example
c = fetch!(raw_counts)
```
The result is a `DataMatrix` (read more about them here: [Data Matrices](@ref)).
Here we have a `DataMatrix` where the rows (variables) are genes and the columns (observations) are cells. Inside, there is a matrix with the raw counts, and annotation tables for the genes and the cells.

Here are the first few cells and genes:
```@repl
c.obs[1:6,:]
c.var[1:6,:]
```

You rarely need to access the raw count matrix directly. It is stored in a blocked format that's efficient for the computations that SingleCellProjections needs. But to take a look, we can convert it to a regular sparse matrix and show a small part of it:
```@example
convert(SparseMatrixCSC, c.matrix)[1:100,1:100]
```


## Quality Filtering

Some of the cells in the data set are of poor quality, and we want to get rid of them before continuing the analysis.
Commonly used quality measures are the fraction of reads from Mitochondrial genes, the total read count per cell, and the number of genes with non-zero counts.
A high fraction of Mitochondrial reads can indicate that a cell is dying. The total read count and number of expressed genes help determine whether there is enough data for a meaningful analysis.

We set up the quality annotations like this:
```@example
counts = Jobs.var_counts_fraction(raw_counts, "fraction_mt", "name"=>startswith("MT-"))
counts = Jobs.var_counts_sum(counts, "total_RNA_count")
counts = Jobs.var_counts_sum(!iszero, counts, "nonzero_RNA_count")
counts = Jobs.obs_counts_sum(!iszero, counts, "nonzero_cell_count")
```

And apply filtering:
```@example
filtered = Jobs.filter_obs("fraction_mt" => <(0.15), counts)
filtered = Jobs.filter_obs("total_RNA_count" => >(1000), filtered)
filtered = Jobs.filter_obs("nonzero_RNA_count" => >(500), filtered)
```

And if we `fetch!` the result:
```@example
fetch!(filtered)
```
We see that filtering reduced the number of cells from 13223 to 11969, and that the new annotations are present in the data matrix.



## Transformation
The raw counts data is not suitable for analyses like PCA, since the data is far from normally distributed.
A common strategy to handle this is to transform the data.
Here we will use [SCTransform](https://github.com/rasmushenningsson/SCTransform.jl) (see also [original sctransform implementation in R](https://github.com/satijalab/sctransform)).
```@example
transformed = Jobs.sctransform(filtered)
fetch!(transformed)
```
From the output, we see that the number of variables has been reduced, since by default, `sctransform` removes variables present in very few cells.

The matrix is now shown as `A+B₁B₂B₃`.
This is normally not very important from the user's point of view, but it is critical for explaining how `SingleCellProjections` can be fast and not use too much memory.
Instead of storing the SCTransformed matrix as a huge dense matrix, it is stored in memory as a `MatrixExpression`, in this case a sparse matrix `A` plus a product of three smaller matrices `B₁`,`B₂` and `B₃`.



## Normalization
After transformation we always want to normalize the data.
At the very least, data should be centered for PCA to work properly. This can be achieved by just running `Jobs.normalize_matrix` with the default parameters.
Here, we also want to regress out `"fraction_mt"`. You can add more `obs` annotations (categorical and/or numerical) to regress out if needed.
```@example
normalized = Jobs.normalize_matrix(transformed, "fraction_mt")
fetch!(normalized)
```
Now the matrix is shown as `A+B₁B₂B₃+(-β)X'`, i.e. another low-rank term was added to handle the normalization/regression.
Since all `ReproducibleJobs.jl` results are read-only, the first two terms can be reused, ensuring that memory is not wasted.


## Principal Component Analysis
Principal Component Analysis (PCA) is commonly used for single cell expression data for two major reasons:

1. It accurately finds a more compact representation of the data. This is important computationally, since it greatly reduces computation time for downstream analyses.
2. It reduces noise by keeping only the common variations in the data set.

```@example
reduced = Jobs.pca(normalized; nsv=40)
fetch!(reduced)
```

Now, the data set has been reduced to 40 variables (the top 40 principal components, ranked by variance explained), and is represented as a single dense matrix.
The cell (observation) annotations are left untouched.


## Annotations
In preparation for the visualization below (or other analyses), we also load some cell-level annotations for our data set.
```@example
annots_path = "annotations/scRNA_AML.tsv"
annots_path = SingleCellDocUtils.get_lilljebjorn_annot_path("scRNA_AML") # hide
annots = Jobs.load_csv(annots_path)
reduced = Jobs.annotate_obs(reduced, annots)
nothing # hide
```



## Visualization
Analyses should generally be performed on the normalized or PCA-reduced data, not on visualization embeddings.
However, for visualization purposes, we want to reduce to 2 or 3 dimensions.


```@raw html
<details>
<summary>Expand this to show some simple example <a href="https://github.com/MakieOrg/Makie.jl">Makie.jl</a> plotting code that is used below to produce the plots.</summary>
```

You can of course use your own favorite plotting library instead.

```@example
using Colors
using WGLMakie

function scatter_3d(job)
    matrix = fetch!(Jobs.get_matrix(job))
    fig = Figure(; size=(768, 768))
    ax = LScene(fig[1, 1])
    scatter!(ax, matrix; color = :black, markersize = 4)
    fig
end

function scatter_categorical_3d(job, annot_name; bg=nothing, colors=nothing)
    matrix = fetch!(Jobs.get_matrix(job))
    annot = fetch!(Jobs.value_column_data(Jobs.annotation(Jobs.get_obs(job), annot_name)))

    fig = Figure(; size=(768, 768))
    ax = LScene(fig[1, 1])

    if bg !== nothing
        bg_matrix = fetch!(Jobs.get_matrix(bg))
        scatter!(ax, bg_matrix; color=colorant"#BFCCE6", markersize=2)
    end

    if colors !== nothing
        unique_annotations = unique(annot)
        unique_annotations_set = Set(unique_annotations)
        colors = filter(x->x[1] in unique_annotations_set, colors)

        categories = first.(colors)
        @assert isempty(setdiff(unique_annotations, categories)) # ensure all categories have colors specified

        plots = [scatter!(ax, matrix[:,isequal.(annot, cat)]; markersize=5, color) for (cat,color) in colors]
    else
        categories = unique(annot)
        plots = [scatter!(ax, matrix[:,isequal.(annot, cat)]; markersize=5) for cat in categories]
    end

    axislegend(ax, plots .=> Ref((;markersize=16)), categories) # use a larger markersize in the legend
    fig
end

```

Use it like this:
```julia
scatter_3d(data)
scatter_categorical_3d(data, "celltype.aml")
```
```@raw html
</details>
```


### Force Layout
To embed the points in 2 or 3 dimensions using a Force Layout (also known as a SPRING plot), we set it up like this:
```@example
fl = Jobs.force_layout(reduced; ndim = 3,
                                seed = 4567,
                                k = 100,
                                k_projection = 25)
scatter_3d(fl)
```

To make a nicer visualization, we use a celltype annotation from the downloaded data to color the plot, and apply a utility function to rotate it such that the most immature cells (Hematopoietic Stem Cells, or HSCs for short) move to the top.
```@example
transform = Jobs.find_optimal_coord_transform(fl, "celltype.aml"=>isequal("HSC"), "celltype.aml"=>isequal("T-cells"), "celltype.aml"=>isequal("B-cells"))
fl = Jobs.transform_coords(fl, transform; keep_var=true)
colors = ["AML Immature" => colorant"#fec44f", "HSC" => colorant"#66b266", "Monocytes" => colorant"#fa9fb5", "GMP" => colorant"#008000", "Megakaryocytic cells" => colorant"#e5687e", "LMPP" => colorant"#756bb1", "Erythroid cells" => colorant"#a7241d", "B-cells" => colorant"#a64ca6", "T-cells" => colorant"#7fb4e5", "NK-cells" => colorant"#196fbe", "Dendritic cells" => colorant"#ff4d00"]
scatter_categorical_3d(fl, "celltype.aml"; colors)
```

### UMAP
```@example
using UMAP
umapped = Jobs.umap(reduced; ndim=3)
scatter_categorical_3d(umapped, "celltype.aml"; colors)
```

### t-SNE
t-SNE is also supported, just run:
```julia
using TSne
tsne_job = Jobs.tsne(reduced; ndim=3)
```


## Projections
`SingleCellProjections` is built to make it very easy to project one dataset onto another. This is useful for comparing new samples against an established reference.

First, we just choose one or more samples to project:
```@example
# Name and location of AML file
proj_name = "AML28"
proj_path = joinpath("samples", string(proj_name, ".h5"))
proj_path = SingleCellDocUtils.get_lilljebjorn_sample_path(proj_name) # hide
proj_raw_counts = Jobs.load_counts(proj_path; sample_names=proj_name)
```

Then, a single call to `Jobs.project` sets up the entire analysis pipeline and projects the AML sample onto the reference map created from the NBM samples.
```@example
proj_fl = Jobs.project(fl, raw_counts=>proj_raw_counts)
scatter_categorical_3d(proj_fl, "celltype.aml"; bg=fl, colors)
```

And here is the projection onto the UMAP embedding of the NBM samples:
```@example
proj_umapped = Jobs.project(umapped, raw_counts=>proj_raw_counts)
scatter_categorical_3d(proj_umapped, "celltype.aml"; bg=umapped, colors)
```
