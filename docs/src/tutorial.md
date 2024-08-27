# Tutorial

```@setup fake
# Mimic the real dataset, without actually loading the data.
# The plots are generated offline and uploaded separately.
# We might consider actually running the tutorial code here - but then we would need to download files, load more packages and to more computations each time we rebuild the docs.


# Useful function to extract a little data from a DataMatrix to use here for show
# function f(col, n; h=10, t=10, filler="")
#   @assert h+t <= n
#   pre = string(col[1:h])
#   mid = string("fill(\"", filler, "\", ", n-h-t, ")")
#   post = string(col[end-t+1:end])
#   string("vcat(", pre, ", ", mid, ", ", post, ')')
# end


using SingleCellProjections, SparseArrays, DataFrames, LinearAlgebra
using UMAP, TSne

function load_counts(args...; sample_names, kwargs...) # Shadow SingleCellProjections.load_counts to fake loading!
    P = 33766
    N = "P1" in sample_names ? 35340 : 42553
    # X = sparse(Int32[], Int32[], Int[], P, N)
    # X = sparse(ones(Int32,N), Int32.(1:N), 1:N, P, N)

    I = repeat(Int32[1,2]; inner=N)
    J = vcat(Int32.(1:N), Int32.(1:N))
    V = vcat(1:N, div.(N:-1:1,3))
    X = sparse(I,J,V,P,N)

    v_id = vcat(["MIR1302-2HG", "FAM138A", "OR4F5", "AL627309.1", "AL627309.3", "AL627309.2", "AL627309.4", "AL732372.1", "OR4F29", "AC114498.1"], string.("dummy_", 1:33746), ["CD169", "CD28", "CD161", "CD163", "CD138-1", "CD164", "CD138-2", "CD144", "CD202b", "CD11c"])
    v_feature_type = vcat(fill("Gene Expression",33538), fill("Antibody Capture", P-33538))

    v_id[33497:33509] .= string.("MT-", 1:13)

    o_cell_id = vcat(["P1_L1_AAACCCAAGACATACA", "P1_L1_AAACCCACATCGGTTA", "P1_L1_AAACCCAGTGGAACAC", "P1_L1_AAACCCATCTGCGGAC", "P1_L1_AAACGAAAGTTACTCG", "P1_L1_AAACGAACAATGAGCG", "P1_L1_AAACGAACACTCCTTG", "P1_L1_AAACGAACAGCATCTA", "P1_L1_AAACGAATCCTCACCA", "P1_L1_AAACGAATCTCACTCG"], string.("dummy_", 1:N-20), ["P2_L5_TTTGGTTGTCCGAAAG", "P2_L5_TTTGGTTTCCTCTAGC", "P2_L5_TTTGGTTTCGTAGGGA", "P2_L5_TTTGGTTTCTTTGATC", "P2_L5_TTTGTTGAGTGTACCT", "P2_L5_TTTGTTGGTACGATCT", "P2_L5_TTTGTTGGTCCTTAAG", "P2_L5_TTTGTTGTCAACACCA", "P2_L5_TTTGTTGTCATGCATG", "P2_L5_TTTGTTGTCCGTGCGA"])
    o_sampleName = vcat(fill("P1",18135), fill("P2", N-18135))
    o_barcode = vcat(["L1_AAACCCAAGACATACA", "L1_AAACCCACATCGGTTA", "L1_AAACCCAGTGGAACAC", "L1_AAACCCATCTGCGGAC", "L1_AAACGAAAGTTACTCG", "L1_AAACGAACAATGAGCG", "L1_AAACGAACACTCCTTG", "L1_AAACGAACAGCATCTA", "L1_AAACGAATCCTCACCA", "L1_AAACGAATCTCACTCG"], string.("dummy_", 1:N-20), ["L5_TTTGGTTGTCCGAAAG", "L5_TTTGGTTTCCTCTAGC", "L5_TTTGGTTTCGTAGGGA", "L5_TTTGGTTTCTTTGATC", "L5_TTTGTTGAGTGTACCT", "L5_TTTGTTGGTACGATCT", "L5_TTTGTTGGTCCTTAAG", "L5_TTTGTTGTCAACACCA", "L5_TTTGTTGTCATGCATG", "L5_TTTGTTGTCCGTGCGA"])

    v = DataFrame(id=v_id, feature_type=v_feature_type, name=v_id, genome="hg19", read="", pattern="", sequence="")
    o = DataFrame(cell_id=o_cell_id, sampleName=o_sampleName, barcode=o_barcode)
    counts = DataMatrix(X, v, o)
end

function sctransform(counts)
    # SingleCellProjections.sctransform(counts; use_cache=false, verbose=false)
    m = SCTransformModel(counts; use_cache=false, verbose=false)
    nvar = 20239
    append!(m.params, m.params[mod1.(1:nvar-size(m.params,1),2),:])
    m.params.id = counts.var.id[1:nvar]
    project(counts, m; verbose=false)
end

svd(args...; nsv, kwargs...) = LinearAlgebra.svd(args...; nsv, subspacedims=nsv, niter=1, kwargs...)
force_layout(args...; kwargs...) = SingleCellProjections.force_layout(reduced; niter=1, kwargs...)

umap(args...; kwargs...) = UMAP.umap(args...; n_epochs=1, init=:random, n_neighbors=2, kwargs...)
tsne(data, d; kwargs...) = TSne.tsne(data, d, 0, 1, 5; verbose=false, progress=false, kwargs...) # We might want to speed this up further by running with fewer cells, takes about half of the total doc generation time
```


For this example we will use PBMC (Peripheral Blood Mononuclear Cell) data from the paper [Integrated analysis of multimodal single-cell data](https://www.sciencedirect.com/science/article/pii/S0092867421005833) by Hao et al.
You can find the original data [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378), in MatrixMarker (.mtx) format.
For convenience, you can [download the samples recompressed as .h5 files](https://github.com/rasmushenningsson/SingleCellExampleData).
Direct links:
* [Cell annotations (.csv.gz)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P.csv.gz)
* [Donor P1 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P1.h5)
* [Donor P2 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P2.h5)
* [Donor P3 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P3.h5)
* [Donor P4 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P4.h5)
* [Donor P5 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P5.h5)
* [Donor P6 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P6.h5)
* [Donor P7 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P7.h5)
* [Donor P8 (.h5)](https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/GSE164378_RNA_ADT_3P_v2/GSE164378_RNA_ADT_3P_P8.h5)

First we load `SingleCellProjections` and the packages `DataFrames` and `CSV` for handling annotations.
```@repl fake
using SingleCellProjections, DataFrames, CSV
```

## Loading Data
Then we load samples "P1" and "P2", by specifiying the paths to the files and naming them.
```@repl fake
base_path = "/path/to/downloads/";

sample_paths = joinpath.(base_path, ["GSE164378_RNA_ADT_3P_P1.h5", "GSE164378_RNA_ADT_3P_P2.h5"]);

counts = load_counts(sample_paths; sample_names=["P1","P2"])
```



Data sets in `SingleCellProjections` are represented as `DataMatrix` objects, which are matrices with annotations for `var` (variables/genes/features) and `obs` (observations, typically cells).
Above, `counts` is a `DataMatrix` where the counts are stored in a sparse matrix.
You can also see the available annotations for variables and observations.
To access the different parts, use:
* `counts.matrix` - For the matrix
* `counts.var` - Variable annotations (`DataFrame`)
* `counts.obs` - Observation annotations (`DataFrame`)


## Cell Annotations
Here we compute a new `obs` annotation where we count the fraction of reads coming from Mitochondrial genes for each cell:
```@repl fake
var_counts_fraction!(counts, "name"=>startswith("MT-"), "feature_type"=>isequal("Gene Expression"), "fraction_mt")
```
Note that the new annotation `fraction_mt` is present in the output.

We will also load some more cell annotations from the provided file.
```julia-repl
julia> cell_annotations = CSV.read(joinpath(base_path, "GSE164378_RNA_ADT_3P.csv.gz"), DataFrame);
```

```@setup fake

counts.obs.fraction_mt[1:10] = [194/5864,606/9333,176/3251,299/4198,343/5486,473/7379,196/4444,174/5693,160/4525,156/3519]

csv_str = """barcode,nCount_ADT,nFeature_ADT,nCount_RNA,nFeature_RNA,orig.ident,lane,donor,time,celltype.l1,celltype.l2,celltype.l3,Phase,Batch
          L1_AAACCCAAGAAACTCA,7535,217,10823,2915,SeuratProject,L1,P2,7,Mono,CD14 Mono,CD14 Mono,G1,Batch1
          L1_AAACCCAAGACATACA,6013,209,5864,1617,SeuratProject,L1,P1,7,CD4 T,CD4 TCM,CD4 TCM_1,G1,Batch1
          L1_AAACCCACAACTGGTT,6620,213,5067,1381,SeuratProject,L1,P4,2,CD8 T,CD8 Naive,CD8 Naive,S,Batch1
          L1_AAACCCACACGTACTA,3567,202,4786,1890,SeuratProject,L1,P3,7,NK,NK,NK_2,G1,Batch1
          L1_AAACCCACAGCATACT,6402,215,6505,1621,SeuratProject,L1,P4,7,CD8 T,CD8 Naive,CD8 Naive,G1,Batch1
          L1_AAACCCACATCAGTCA,5297,212,4332,1633,SeuratProject,L1,P3,2,CD8 T,CD8 TEM,CD8 TEM_1,G1,Batch1
          L1_AAACCCACATCGGTTA,7634,219,9333,2672,SeuratProject,L1,P1,7,Mono,CD16 Mono,CD16 Mono,G1,Batch1
          L1_AAACCCACATGGATCT,8210,222,3589,1122,SeuratProject,L1,P4,2,B,B intermediate,B intermediate lambda,G1,Batch1
          L1_AAACCCAGTGGAACAC,2847,201,3251,1375,SeuratProject,L1,P1,2,NK,NK,NK_2,G2M,Batch1
          L1_AAACCCATCCACACCT,4557,209,3401,1200,SeuratProject,L1,P3,2,CD8 T,CD8 Naive,CD8 Naive,S,Batch1
          L1_AAACCCATCTGCGGAC,5129,212,4198,1318,SeuratProject,L1,P1,0,CD4 T,CD4 TCM,CD4 TCM_1,S,Batch1
          L1_AAACGAAAGTTACTCG,7630,208,5486,1390,SeuratProject,L1,P1,0,CD4 T,CD4 TCM,CD4 TCM_3,G1,Batch1
          """
cell_annotations = CSV.read(codeunits(csv_str), DataFrame)
```

To merge, we use the `DataFrames` function `leftjoin!`, since it takes care of matching the cells in `counts` to the cells in `cell_annotations` based on the `:barcode` column.
```@repl fake
leftjoin!(counts.obs, cell_annotations; on=:barcode);
```

```@setup fake
counts.obs[34639+1:end,"celltype.l1"] .= "other"
```

Let's look at some annotations for the first few cells:
```@repl fake
counts.obs[1:5,["cell_id","sampleName","barcode","fraction_mt","celltype.l1"]]
```

## Transformation
The raw counts data is not suitable for analyses like PCA, since the data is far from normally distributed.
A common strategy to handle this is to transform the data.
Here we will use [SCTransform](https://github.com/rasmushenningsson/SCTransform.jl) (see also [original sctransform implementation in R](https://github.com/satijalab/sctransform)).
```@repl fake
transformed = sctransform(counts)
```
From the output, we see that the number of variables have been reduced, since the default `sctransform` options remove variables present in very few cells and only keeps variables with `feature_type` set to `"Gene Expression"`.

The matrix is now shown as `A+B₁B₂B₃`.
This is normally not very important from the user's point of view, but it is critical for explaining how `SingleCellProjections` can be fast and not use too much memory.
Instead of storing the SCTransformed matrix as a huge dense matrix, it is stored in memory as a `MatrixExpression`, in this case a sparse matrix `A` plus a product of three smaller matrices `B₁`,`B₂` and `B₃`.


## Normalization
After transformation we always want to normalize the data.
At the very least, data should be centered for PCA to work properly, this can be achieved by just running `normalize_matrix` with the default parameters.
Here, we also want to regress out `"fraction_mt"`. You can add more `obs` annotations (categorical and/or numerical) to regress out if you need.
```@repl fake
normalized = normalize_matrix(transformed, "fraction_mt")
```
Now the matrix is shown as `A+B₁B₂B₃+(-β)X'`, i.e. another low-rank term was added to handle the normalization/regression.
The first two terms are reused to make sure memory is not wasted.

## Filtering
It is possible to filter variables and observations.
Here we keep all cells that are not labeled as `"other"`.
```@repl fake
filtered = filter_obs("celltype.l1"=>!isequal("other"), normalized)
```

## Principal Component Analysis (PCA)
Now we are ready to perform Principal Component Analysis (PCA).
This is computed by the Singular Value Decomposition (SVD), so we should call the `svd` function.
The number of dimensions is specified using the `nsv` parameter.
```@repl fake
reduced = svd(filtered; nsv=20)
```
The matrix is now stored as an `SVD` object, which includes low dimensional representations of the observations and variables.
To retrieve the low dimensional coordinates, use `obs_coordinates` and `var_coordinates` respectively.

![Principal Component Analysis](https://user-images.githubusercontent.com/16546530/228492439-7e31d4d1-1edd-493e-9e96-48fd2583b93d.svg)

[Download interactive PCA plot](https://github.com/rasmushenningsson/SingleCellProjections.jl/files/11099039/svd.zip).


## Visualization
```@raw html
<details>
<summary>Expand this to show some example PlotlyJS plotting code.</summary>
```

You can of course use your own favorite plotting library instead.
Use `obs_coordinates` to get the coordinates for each cell, and `data.obs` to access cell annotations for coloring.

```julia
using PlotlyJS
function plot_categorical_3d(data, annotation; marker_size=3)
    points = obs_coordinates(data)
    traces = GenericTrace[]
    for sub in groupby(data.obs, annotation; sort=true)
        value = sub[1,annotation]
        ind = parentindices(sub)[1]
        push!(traces, scatter3d(;x=points[1,ind], y=points[2,ind], z=points[3,ind], mode="markers", marker_size, name=value))
    end
    plot(traces, Layout(;legend=attr(itemsizing="constant")))
end
```

Use it like this:
```julia-repl
julia> plot_categorical_3d(reduced, "celltype.l1")
```
```@raw html
</details>
<br>
```

For visualization purposes, it is often useful to further reduce the dimension after running PCA.
(In contrast, analyses are generally run on the PCA/normalized/original data, since the methods below necessarily distort the data to force it down to 2 or 3 dimensions.)


### Force Layout
Force Layout plots (also known as SPRING Plots) are created like this:
```@repl fake
fl = force_layout(reduced; ndim=3, k=100)
```
![force_layout](https://user-images.githubusercontent.com/16546530/228492990-14c31888-28e1-4f3c-8062-f10682e55430.svg)

[Download interactive Force Layout plot](https://github.com/rasmushenningsson/SingleCellProjections.jl/files/11099049/force_layout.zip).



### UMAP
`SingleCellProjections` can be used together with [UMAP.jl](https://github.com/dillondaudert/UMAP.jl):
```@repl fake
using UMAP

umapped = umap(reduced, 3)
```
![umap](https://user-images.githubusercontent.com/16546530/228493039-588cafde-86fc-4fae-ae58-fa65cf59e929.svg)

[Download interactive UMAP plot](https://github.com/rasmushenningsson/SingleCellProjections.jl/files/11099051/umap.zip).


### t-SNE
Similarly, t-SNE plots are supported using [TSne.jl](https://github.com/lejon/TSne.jl).
In this example, we just run it one every 10ᵗʰ cell, because t-SNE doesn't scale very well with the number of cells:
```@repl fake
using TSne

t = tsne(reduced[:,1:10:end], 3)
```
![t-SNE](https://user-images.githubusercontent.com/16546530/228493090-06cc9f32-4e11-4441-a9b3-8da93d503a83.svg)

[Download interactive t-SNE plot](https://github.com/rasmushenningsson/SingleCellProjections.jl/files/11099055/t-SNE.zip).


### Other
It is of course possible to use your own favorite dimension reduction method/package.
The natural input for most cases are the coordinates after dimension reduction by PCA (`obs_coordinates(reduced)`).


## Projections
`SingleCellProjections` is built to make it very easy to project one dataset onto another.

Let's load count data for two more samples:
```@repl fake
sample_paths_proj = joinpath.(base_path, ["GSE164378_RNA_ADT_3P_P5.h5", "GSE164378_RNA_ADT_3P_P6.h5"]);

counts_proj = load_counts(sample_paths_proj; sample_names=["P5","P6"]);

leftjoin!(counts_proj.obs, cell_annotations; on=:barcode);

counts_proj
```

```@setup fake
counts_proj.obs[41095+1:end,"celltype.l1"] .= "other"
```


And project them onto the Force Layout we created above:
```@repl fake
fl_proj = project(counts_proj, fl)
```
The result looks similar to the force layout plot above, since the donors "P5" and "P6" are similar to donors "P1" and "P2".

![force_layout_projected](https://user-images.githubusercontent.com/16546530/228493122-bfdee02b-e6fb-4219-88d1-31ab4e323ca9.svg)

[Download interactive Force Layout projection plot](https://github.com/rasmushenningsson/SingleCellProjections.jl/files/11099059/force_layout_projected.zip).

Under the hood, `SingleCellProjections` recorded a `ProjectionModel` for every step of the analysis leading up to the Force Layout.
Let's take a look:
```@repl fake
fl.models
```
When projecting, these models are applied one by one (C.f. output from `project` above), ensuring that the projected data is processed correctly.
In most cases, projecting is **not** the same as running the same analysis independently, since information about the data set is recorded in the model.
