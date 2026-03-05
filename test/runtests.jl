using Test
using SingleCellProjections
using SingleCellProjections: Projectable, ProjectOnto, Action, DataMatrixFunction, DataMatrixField, DataMatrixFieldFunction, Mat, Var, Obs, MatFunction, VarFunction, ObsFunction, get_matrix_spec
import .SingleCellProjectionsCore as SCPCore
using .SingleCellProjectionsCore.MatrixExpressions
using SCTransform
using ReproducibleJobs: ReproducibleJobs, Scheduler, TimestampedFilePath, get_scheduler, with_scheduler, fetch!, forward, forward_once, create_spec, Job, Preprocess

using StableRNGs

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

using CSV

include("test_utils.jl")
include("common_data.jl")


# mktempdir() do tmp # Cleanup directly
let tmp = mktempdir() # Cleanup when Julia process exits - useful for inspecting
	@testset "SingleCellProjections.jl" begin
		scheduler = Scheduler(; dir=tmp)
		with_scheduler(scheduler) do

			# Consider doing this kind of cache testing in ReproducibleJobs.jl only.
			@testset "$cache_status" for cache_status in ("New Disk Cache", "Reused Disk Cache")
			# @testset "$cache_status" for cache_status in ("New Disk Cache",)
				# TODO: Find a better way to do this (probably similar to how we replace and reset the Cache above)

				empty!(scheduler)

				include("projectables.jl")
				include("tables.jl")
				include("load.jl")
				include("transform.jl")
				include("reduce.jl")
				include("filter.jl")
				include("subset.jl")
			end
		end
	end

end

include("SingleCellProjectionsCore/runtests.jl")
include("MatrixExpressions/runtests.jl")
