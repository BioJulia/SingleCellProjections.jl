module SingleCellProjections

export
	SingleCellProjectionsCore, # TODO: make public instead
	DataMatrix,
	Jobs # TODO: remove probably

# This symbol is only defined on Julia versions that support extensions
isdefined(Base, :get_extension) || using Requires

include("SingleCellProjectionsCore/SingleCellProjectionsCore.jl")

import .SingleCellProjectionsCore as SCPCore
using .SCPCore: DataMatrix
import SCTransform
import SingleCell10x
using DataFrames
import StableHashTraits
import LinearAlgebra

using ReproducibleJobs
using ReproducibleJobs: create_spec, ReadOnly, SpecArgs, ChecksummedFilePath, checksummedfilepath_job, ifelse_spec

ReproducibleJobs.unmanage_rec(x::SCPCore.ValueVector) = x
ReproducibleJobs.unmanage_rec(x::SCPCore.ProjectionModel) = x


# TODO: This is a temporary solution when refactoring, remove
module Jobs
	function load_counts end
	function get_matrix end
	function get_var end
	function get_obs end
	function annotate end
	function annotate_var end
	function annotate_obs end
	function var_counts_fraction end
	function var_counts_sum end
	function obs_counts_fraction end
	function obs_counts_sum end
	function find_matching_ids end
	function subset_annotation end
	function subset_var end
	function subset_obs end
	function subset_matrix end
	function filter_annotations end
	function filter_var end
	function filter_obs end
	function filter_matrix end
	function sctransform end
	function logtransform end
	function tf_idf_transform end
	function center_matrix end # TEMP
	function designmatrix end
	function designmatrix2 end # TEMP
	function negative_regression_matrix end
	function normalize_matrix end
	function svd end
	function pca end
	function loadings end
	function force_layout end
	function transpose end
	function umap end
	function tsne end
	function project end
end



include("types.jl")
include("projectables.jl")
include("datamatrixfunctions.jl")
include("internal.jl")
include("matrix_arithmetic.jl")
include("load.jl")
include("annotate.jl")
include("filter.jl")
include("transform.jl")
include("design.jl")
include("normalize.jl")
include("reduce.jl")
include("adjoint.jl")


# include("precompile.jl")



@static if !isdefined(Base, :get_extension)
	function __init__()
		@require UMAP="c4f8c510-2410-5be4-91d7-4fbaeb39457e" include("../ext/SingleCellProjectionsUMAPExt.jl")
		@require TSne="24678dba-d5e9-5843-a4c6-250288b04835" include("../ext/SingleCellProjectionsTSneExt.jl")
		@require PrincipalMomentAnalysis="6a3ba550-3b7f-11e9-2734-d9178ad1e8db" include("../ext/SingleCellProjectionsPrincipalMomentAnalysisExt.jl")
		@require Muon="446846d7-b4ce-489d-bf74-72da18fe3629" include("../ext/SingleCellProjectionsMuonExt.jl")
	end
end



end
