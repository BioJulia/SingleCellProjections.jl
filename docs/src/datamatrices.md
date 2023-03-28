# Data Matrices


```@setup data
# Mimic the dataset from the tutorial without including it in the docs building


# Useful function to extract a little data from a DataMatrix to use here for show
# function f(col, n; h=10, t=10, filler="")
# 	@assert h+t <= n
# 	pre = string(col[1:h])
# 	mid = string("fill(\"", filler, "\", ", n-h-t, ")")
# 	post = string(col[end-t+1:end])
# 	string("vcat(", pre, ", ", mid, ", ", post, ')')
# end



using SingleCellProjections, SparseArrays, DataFrames
P,N = 33766,35340
X = sparse(Int32[], Int32[], Int[], P, N)

v_id = vcat(["MIR1302-2HG", "FAM138A", "OR4F5", "AL627309.1", "AL627309.3", "AL627309.2", "AL627309.4", "AL732372.1", "OR4F29", "AC114498.1"], string.("dummy_", 1:33746), ["CD169", "CD28", "CD161", "CD163", "CD138-1", "CD164", "CD138-2", "CD144", "CD202b", "CD11c"])
v_feature_type = vcat(fill("Gene Expression",33538), fill("Antibody Capture", P-33538))

o_id = vcat(["P1_L1_AAACCCAAGACATACA", "P1_L1_AAACCCACATCGGTTA", "P1_L1_AAACCCAGTGGAACAC", "P1_L1_AAACCCATCTGCGGAC", "P1_L1_AAACGAAAGTTACTCG", "P1_L1_AAACGAACAATGAGCG", "P1_L1_AAACGAACACTCCTTG", "P1_L1_AAACGAACAGCATCTA", "P1_L1_AAACGAATCCTCACCA", "P1_L1_AAACGAATCTCACTCG"], string.("dummy_", 1:35320), ["P2_L5_TTTGGTTGTCCGAAAG", "P2_L5_TTTGGTTTCCTCTAGC", "P2_L5_TTTGGTTTCGTAGGGA", "P2_L5_TTTGGTTTCTTTGATC", "P2_L5_TTTGTTGAGTGTACCT", "P2_L5_TTTGTTGGTACGATCT", "P2_L5_TTTGTTGGTCCTTAAG", "P2_L5_TTTGTTGTCAACACCA", "P2_L5_TTTGTTGTCATGCATG", "P2_L5_TTTGTTGTCCGTGCGA"])
o_sampleName = vcat(fill("P1",18135), fill("P2", N-18135))
o_barcode = vcat(["L1_AAACCCAAGACATACA", "L1_AAACCCACATCGGTTA", "L1_AAACCCAGTGGAACAC", "L1_AAACCCATCTGCGGAC", "L1_AAACGAAAGTTACTCG", "L1_AAACGAACAATGAGCG", "L1_AAACGAACACTCCTTG", "L1_AAACGAACAGCATCTA", "L1_AAACGAATCCTCACCA", "L1_AAACGAATCTCACTCG"], string.("dummy_", 1:35320), ["L5_TTTGGTTGTCCGAAAG", "L5_TTTGGTTTCCTCTAGC", "L5_TTTGGTTTCGTAGGGA", "L5_TTTGGTTTCTTTGATC", "L5_TTTGTTGAGTGTACCT", "L5_TTTGTTGGTACGATCT", "L5_TTTGTTGGTCCTTAAG", "L5_TTTGTTGTCAACACCA", "L5_TTTGTTGTCATGCATG", "L5_TTTGTTGTCCGTGCGA"])

v = DataFrame(id=v_id, feature_type=v_feature_type, name=v_id, genome="hg19", read="", pattern="", sequence="")
o = DataFrame(id=o_id, sampleName=o_sampleName, barcode=o_barcode)
data = DataMatrix(X, v, o)
```


`DataMatrix` objects -- annotated matrices where rows are variables and columns are observations -- are central in SingleCellProjections.jl.
A `DataMatrix` is also sometimes called an "Assay", in other software packages.

An overview of a `DataMatrix` is shown when the object is displayed:
```@repl data
data
```
Here we see the matrix size (number of variables and observations), a brief description of the matrix contents, and an overview of available variable and observation annotations. The underlined annotation names are the ID columns (see [IDs](@ref) below for more details).



## Variables
Variables, or `var` for short, are typically genes, features (such as CITE-seq features) or variables after dimension reduction (e.g. "UMAP1").
The variables are stored as a [`DataFrame`](https://dataframes.juliadata.org/stable/) and can be accessed by:
```@repl data
data.var
```


## Observations
Observations, or `obs` for short, are typically cells, but can in theory be any kind of observation.
The observations are stored as a [`DataFrame`](https://dataframes.juliadata.org/stable/) and can be accessed by:
```@repl data
data.obs
```


## IDs
Each variable and each observation must have a unique ID, that is, each row in the `DataFrame` should be unique if we consider the ID columns only.
As seen above, the ID columns are underlined when displaying a DataMatrix.
We can also access them directly:
```@repl data
data.var_id_cols
```
```@repl data
data.obs_id_cols
```

Most of the time, IDs are handled automatically by SingleCellProjections.jl.
Sometimes, you need to make sure IDs are unique when loading or merging data matrices.
In particular, when loading a `DataMatrix` that should be projected onto another `DataMatrix`, the user must ensure that relevant IDs are matching.



## Matrix
The matrix can be accessed by `data.matrix`.
Depending on the stage of analysis, different kinds of matrices (or matrix-like objects) are used.
Most of this complexity is hidden from the user, but internally SingleCellProjections.jl depends on this functionality to be fast and to reduce memory usage.

!!! warning "Read-only"
    SingleCellProjections.jl will reuse matrices when possible, in order to reduce memory usage.
    E.g. [`normalize_matrix`](@ref) will reuse and extend the Matrix Expression of the source `DataMatrix`, without creating a copy of the actual data.
    When matrices are reused/copied is considered an implementation detail, and can change at any time.
    Users of SingleCellProjections.jl should thus consider the matrices to be "read-only".
    This should rarely present problems in practice.


Roughly, the matrix types used at different stages are:

1. Counts - [`SparseMatrixCSC`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
2. Transformed and normalized data - [Matrix Expressions](@ref)
3. SVD (PCA) result - [`SVD`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SVD)
4. ForceLayout/UMAP/t-SNE result - `Matrix{Float64}`


