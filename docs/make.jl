using SingleCellProjections
using Documenter

DocMeta.setdocmeta!(SingleCellProjections, :DocTestSetup, :(using SingleCellProjections); recursive=true)

makedocs(;
    modules=[SingleCellProjections],
    authors="Rasmus Henningsson <rasmus.henningsson@med.lu.se>",
    repo="https://github.com/rasmushenningsson/SingleCellProjections.jl/blob/{commit}{path}#{line}",
    sitename="SingleCellProjections.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rasmushenningsson.github.io/SingleCellProjections.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Interface" => "interface.md",
    ],
)

deploydocs(;
    repo="github.com/rasmushenningsson/SingleCellProjections.jl",
    devbranch="main",
)
