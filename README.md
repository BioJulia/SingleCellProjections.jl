# SingleCellProjections.jl
SingleCellProjections.jl is an easy to use and powerful package for analysis of Single Cell Expression data in Julia.
It is faster and uses less memory than existing solutions since the data is internally represented as expressions of sparse and low rank matrices, instead of storing huge dense matrices.
In particular, it efficiently performs PCA (Principal Component Analysis), a natural starting point for downstream analysis, and supports both standard workflows and projections onto a base data set.

## Installation
SingleCellProjections.jl will soon be registered in the Julia General Registry.
In the meantime, you can install it directly from GitHub:

```julia
using Pkg;
Pkg.add(url="https://github.com/rasmushenningsson/SCTransform.jl.git")
Pkg.add(url="https://github.com/rasmushenningsson/SingleCellProjections.jl.git")
```

## Threading
SingleCellProjections.jl heavily relies on threading. Please make sure to [enable threading in Julia](https://docs.julialang.org/en/v1/manual/multi-threading/) to dramatically improve computation speed.
