using Documenter
using Artifacts
using LazyArtifacts

using SingleCellProjections
using SparseArrays # helps to remove "SparseArrays." when printing @repl/@example blocks
using UMAP, TSne # To get docstrings for glue code
using WGLMakie
using Bonito

# For consistency in printing (for DataFrames in particular)
ENV["COLUMNS"] = 100
ENV["LINES"] = 16

module SingleCellDocUtils
	using Pkg.Artifacts
	using LazyArtifacts

	function _lilljebjorn_dir(name)
		ap = joinpath(@__DIR__, "Artifacts.toml")
		artifact_name = "Lilljebjorn2025_$name"
		ensure_artifact_installed(artifact_name, ap)
		p = artifact_path(artifact_hash(artifact_name, ap))
	end


	# On system with good symlink support (everything except Windows), we setup symlinks to make the docs display nicer paths. :)
	function _get_lilljebjorn_file_path(name, dir, extension)
		@static if !Sys.iswindows()
			link_name = joinpath(dir, string(name, '.', extension))
			isfile(link_name) && return link_name
			isdir(dir) || mkdir(dir)
		end
		p = joinpath(_lilljebjorn_dir(name), string(name, '.', extension))
		@static if !Sys.iswindows()
			symlink(p, link_name)
			link_name
		else
			p
		end
	end


	get_lilljebjorn_sample_path(name) = _get_lilljebjorn_file_path(name, "samples", "h5")
	get_lilljebjorn_annot_path(name) = _get_lilljebjorn_file_path(name, "annotations", "tsv")
end


DocMeta.setdocmeta!(SingleCellProjections, :DocTestSetup, :(using SingleCellProjections); recursive=true)




makedocs(;
	modules = [SingleCellProjections],
	checkdocs_ignored_modules = [SingleCellProjections.SingleCellProjectionsCore],
	authors = "Rasmus Henningsson <rasmus.henningsson@med.lu.se>",
	repo = Remotes.GitHub("BioJulia", "SingleCellProjections.jl"),
	sitename = "SingleCellProjections.jl",
	format = Documenter.HTML(;
		prettyurls = get(ENV, "CI", "false") == "true",
		canonical = "https://BioJulia.github.io/SingleCellProjections.jl",
		edit_link = "main",
		assets = String[],
		ansicolor = true, # for underlining to work in REPL output
	),
	pages=[
		"Home" => "index.md",
		"Tutorial" => "tutorial.md",
		"User Guide" => "userguide.md",
		"Data Matrices" => "datamatrices.md",
		"Matrix Expressions" => "matrixexpressions.md",
		"Interface" => "interface.md",
	],
	# pagesonly = true, # This restricts doc generation to the md files provided above
)

deploydocs(;
	repo="github.com/BioJulia/SingleCellProjections.jl",
	devbranch="main",
)
