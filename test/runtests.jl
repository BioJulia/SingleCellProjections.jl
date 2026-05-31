using Test
using SingleCellProjections
using SingleCellProjections: Projectable, ProjectOnto, Action, DataMatrixFunction, DataMatrixField, DataMatrixFieldFunction, Mat, Var, Obs, MatFunction, VarFunction, ObsFunction, get_matrix_spec
import .SingleCellProjectionsCore as SCPCore
using .SingleCellProjectionsCore.MatrixExpressions
using .SCPCore: unblockify
using SCTransform
using ReproducibleJobs: ReproducibleJobs, Scheduler, TimestampedFilePath, get_scheduler, with_scheduler, fetch!, forward!, forward_once!, create_spec, Preprocess

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
# using PrincipalMomentAnalysis

using CSV

include("test_utils.jl")
include("common_data.jl")


# mktempdir() do tmp # Cleanup directly
let tmp = mktempdir() # Cleanup when Julia process exits - useful for inspecting
	@testset "SingleCellProjections.jl" begin
		# The 2nd time we run, the on-disk cache is reused. (Consider doing this kind of cache testing in ReproducibleJobs.jl only.)
		@testset "$cache_status" for cache_status in ("New Disk Cache", "Reused Disk Cache")
		# @testset "$cache_status" for cache_status in ("New Disk Cache",)
			with_scheduler(Scheduler(; dir=tmp)) do
				include("projectables.jl")
				include("tables.jl")
				include("load.jl")
				include("transform.jl")
				include("reduce.jl")
				include("filter.jl")
				include("subset.jl")
				include("sum_squared.jl")
			end
		end
	end
end

include("SingleCellProjectionsCore/runtests.jl")
include("MatrixExpressions/runtests.jl")
