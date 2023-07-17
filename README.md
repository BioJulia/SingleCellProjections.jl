# SingleCellProjections.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://rasmushenningsson.github.io/SingleCellProjections.jl/dev/)
[![Build Status](https://github.com/rasmushenningsson/SingleCellProjections.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rasmushenningsson/SingleCellProjections.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rasmushenningsson/SingleCellProjections.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rasmushenningsson/SingleCellProjections.jl)


SingleCellProjections.jl is an easy to use and powerful package for analysis of Single Cell Expression data in Julia.
It is faster and uses less memory than existing solutions since the data is internally represented as expressions of sparse and low rank matrices, instead of storing huge dense matrices.
In particular, it efficiently performs PCA (Principal Component Analysis), a natural starting point for downstream analysis, and supports both standard workflows and projections onto a base data set.


## Installation
Install SingleCellProjections.jl by running the following commands in Julia:

```julia
using Pkg
Pkg.add("SingleCellProjections")
```


## Threading
SingleCellProjections.jl relies heavily on threading. Please make sure to [enable threading in Julia](https://docs.julialang.org/en/v1/manual/multi-threading/) to dramatically improve computation speed.


## Tutorial
Here is a [tutorial](https://rasmushenningsson.github.io/SingleCellProjections.jl/dev/tutorial/), showcasing SingleCellProjections.jl functionality using a PBMC (Peripheral Blood Mononuclear Cell) data set.

![force_layout](https://user-images.githubusercontent.com/16546530/228492990-14c31888-28e1-4f3c-8062-f10682e55430.svg)

Force layout plot of the PBMC data.

## Documentation
For more information, please refer to the [documentation](https://rasmushenningsson.github.io/SingleCellProjections.jl/dev/).
