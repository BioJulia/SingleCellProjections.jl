# SingleCellProjections.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://biojulia.dev/SingleCellProjections.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://biojulia.dev/SingleCellProjections.jl/dev/)
[![Build Status](https://github.com/BioJulia/SingleCellProjections.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BioJulia/SingleCellProjections.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/BioJulia/SingleCellProjections.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BioJulia/SingleCellProjections.jl)


Please note SingleCellProjections v0.5 has completely changed the public interface, with major improvements. Refer to the [tutorial](https://biojulia.dev/SingleCellProjections.jl/dev/tutorial/) for how to use the new interface.


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
Here is a [tutorial](https://biojulia.dev/SingleCellProjections.jl/dev/tutorial/), showcasing SingleCellProjections.jl functionality using an AML (Acute Myeloid Leukemia) data set.


## Documentation
For more information, please refer to the [documentation](https://biojulia.dev/SingleCellProjections.jl/dev/).


## Example plots

Here are some example plots using an [Acute Myeloid Leukemia (AML)](https://en.wikipedia.org/wiki/Acute_myeloid_leukemia) data set from the paper:
> Henrik Lilljebjörn, Pablo Peña-Martínez, Hanna Thorsson, Rasmus Henningsson, Marianne Rissler, Niklas Landberg, Noelia Puente-Moncada, Sofia von Palffy, Vendela Rissler, Petr Stanek, Jonathan Desponds, Xiangfu Zhong, Gunnar Juliusson, Vladimir Lazarevic, Sören Lehmann, Magnus Fontes, Helena Ågerstam, Carl Sandén, Christina Orsmark-Pietras, Thoas Fioretos. "[The AML cellular state space unveils *NPM1* immune evasion subtypes with distinct clinical outcomes](https://doi.org/10.1038/s41467-025-66546-6)". Nat Commun 16, 10592 (2025).


A force layout plot of all the NBM (Normal Bone Marrow) samples:
![NBM force layout](https://github.com/user-attachments/assets/dc0c8ca1-7ada-419b-b336-5724f3b2bc92)

A projection of a single sample onto the NBM force layout:
![Projection scatter plot](https://github.com/user-attachments/assets/241781b2-cb0f-4c13-b7b7-325a4a1c7e15)

The same projection, but visualized in a contour plot (the numbers on the contour lines show density in number of cells per grid square):
![Projection contour plot](https://github.com/user-attachments/assets/4d0e7968-cbba-41ce-81b0-107410fb9281)
