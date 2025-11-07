using Test
using SingleCellProjections
using .SingleCellProjections.SingleCellProjectionsCore
using .SingleCellProjectionsCore.MatrixExpressions
using ReproducibleJobs: ReproducibleJobs, Cache, get_cache, with_cache, fetch!


using LinearAlgebra
using SparseArrays
using HypothesisTests
using GLM: GLM, StatsModels, lm, coeftable, modelmatrix
using DelimitedFiles
using DataFrames
using CodecZlib

using UMAP
using TSne
using PrincipalMomentAnalysis

include("test_utils.jl")
include("common_data.jl")


# mktempdir() do tmp # Cleanup directly
let tmp = mktempdir() # Cleanup when Julia process exits - useful for inspecting

	# TODO: Find a better way to do this (probably similar to how we replace and reset the Cache below)
	empty!(ReproducibleJobs.default_scheduler())
	empty!(ReproducibleJobs.default_deduplicator().d)

	with_cache(Cache(tmp)) do
		@testset "SingleCellProjections.jl" begin
			include("load.jl")
			include("transform.jl")
			include("filter.jl")
		end
	end
end

include("SingleCellProjectionsCore/runtests.jl")
include("MatrixExpressions/runtests.jl")
