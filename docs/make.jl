using SingleCellProjections
using SparseArrays # helps to remove "SparseArrays." when printing @repl/@example blocks
using UMAP, TSne # To get docstrings for glue code
using Documenter

# For consistency in printing (for DataFrames in particular)
ENV["COLUMNS"] = 100
ENV["LINES"] = 16

DocMeta.setdocmeta!(SingleCellProjections, :DocTestSetup, :(using SingleCellProjections); recursive=true)

makedocs(;
    modules=[SingleCellProjections],
    authors="Rasmus Henningsson <rasmus.henningsson@med.lu.se>",
    repo="https://github.com/BioJulia/SingleCellProjections.jl/blob/{commit}{path}#{line}",
    sitename="SingleCellProjections.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://BioJulia.github.io/SingleCellProjections.jl",
        edit_link="main",
        assets=String[],
        ansicolor = true, # for underlining to work in REPL output
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Data Matrices" => "datamatrices.md",
        "Matrix Expressions" => "matrixexpressions.md",
        "Interface" => "interface.md",
    ],
)

deploydocs(;
    repo="github.com/BioJulia/SingleCellProjections.jl",
    devbranch="main",
)
