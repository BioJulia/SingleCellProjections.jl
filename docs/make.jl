using Documenter
using Artifacts
using LazyArtifacts

using SingleCellProjections
using SparseArrays # helps to remove "SparseArrays." when printing @repl/@example blocks
using WGLMakie
using Bonito

# For consistency in printing (for DataFrames in particular)
ENV["COLUMNS"] = 100
ENV["LINES"] = 16

module SingleCellDocUtils
	using Pkg.Artifacts
	using LazyArtifacts

	# Symlinks we create in the Documenter workdir so the docs can show short, tidy paths. They
	# point into the Julia artifact depot (absolute targets), so they must be removed before
	# deploying — otherwise GitHub Pages' build resolves realpaths and fails on the dangling
	# links. Keys are absolute link paths (the workdir CWD at creation time differs from make.jl's
	# CWD at cleanup time); values are the intended targets, kept for diagnostics.
	const _created_links = Dict{String,String}()

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
			_created_links[abspath(link_name)] = p
			link_name
		else
			p
		end
	end

	# Remove the symlinks created above. Verify each is really a symlink before deleting, so we
	# never touch a real file if something unexpected is on disk.
	function cleanup_links()
		for (link, target) in _created_links
			if islink(link)
				rm(link)
			else
				@warn "Expected a symlink; leaving it in place" link target
			end
		end
		empty!(_created_links)
	end


	get_lilljebjorn_sample_path(name) = _get_lilljebjorn_file_path(name, "samples", "h5")
	get_lilljebjorn_annot_path(name) = _get_lilljebjorn_file_path(name, "annotations", "tsv")
end


DocMeta.setdocmeta!(SingleCellProjections, :DocTestSetup, :(using SingleCellProjections); recursive=true)




makedocs(;
	modules = [SingleCellProjections],
	checkdocs_ignored_modules = [SingleCellProjections.SCPCore, SingleCellProjections.Impl],
	authors = "Rasmus Henningsson <rasmus.henningsson@med.lu.se>",
	repo = Remotes.GitHub("BioJulia", "SingleCellProjections.jl"),
	sitename = "SingleCellProjections.jl",
	format = Documenter.HTML(;
		prettyurls = get(ENV, "CI", "false") == "true",
		canonical = "https://BioJulia.github.io/SingleCellProjections.jl",
		edit_link = "main",
		assets = String[],
		ansicolor = true, # for underlining to work in REPL output
		size_threshold_ignore = ["tutorial.md"], # interactive WGLMakie plots inline a lot of data
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

# Drop the artifact-depot symlinks before deploying (see SingleCellDocUtils above): they are only
# needed while the @example blocks run, and publishing dangling absolute links breaks the GitHub
# Pages build.
SingleCellDocUtils.cleanup_links()

deploydocs(;
	repo="github.com/BioJulia/SingleCellProjections.jl",
	devbranch="main",
)
