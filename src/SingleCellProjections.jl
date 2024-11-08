module SingleCellProjections

export
	DataMatrix,
	ProjectionModel,
	FilterModel,
	LogTransformModel,
	TFIDFTransformModel,
	SCTransformModel,
	NormalizationModel,
	SVDModel,
	NearestNeighborModel,
	ObsAnnotationModel,
	VarCountsFractionModel,
	VarCountsSumModel,
	PseudoBulkModel,
	project,
	set_var_id_col!,
	set_obs_id_col!,
	var_coordinates,
	obs_coordinates,
	load10x,
	loadh5ad,
	load_counts,
	merge_counts,
	filter_matrix,
	filter_var,
	filter_obs,
	covariate,
	designmatrix,
	normalize_matrix,
	svd,
	update_matrix,
	logtransform,
	sctransform,
	tf_idf_transform,
	knn_adjacency_matrix,
	force_layout,
	var_to_obs!,
	var_to_obs,
	var_to_obs_table,
	var_counts_fraction!,
	var_counts_fraction,
	var_counts_sum!,
	var_counts_sum,
	pseudobulk,
	local_outlier_factor!,
	local_outlier_factor,
	local_outlier_factor_table,
	local_outlier_factor_projection!,
	local_outlier_factor_projection,
	local_outlier_factor_projection_table,
	ftest!,
	ftest,
	ftest_table,
	ttest!,
	ttest,
	ttest_table,
	mannwhitney!,
	mannwhitney,
	mannwhitney_table,
	create_datamatrix,
	create_var,
	create_obs

using LinearAlgebra
import LinearAlgebra: svd

using SparseArrays
using Statistics

using HDF5, H5Zblosc
using DataFrames

using Missings

using Random

using NearestNeighbors
using StaticArrays

using Distributions

import SCTransform: SCTransform, scparams, sctransform
using SingleCell10x


include("MatrixExpressions/MatrixExpressions.jl")
using .MatrixExpressions


# This symbol is only defined on Julia versions that support extensions
isdefined(Base, :get_extension) || using Requires


include("random.jl")
include("utils.jl")
include("table_utils.jl")
include("threaded_sparse_row_map.jl")

include("bilinear.jl")
include("sctransformsparse.jl")

include("implicitsvd.jl")

include("annotations.jl")
include("annotation_utils.jl")

include("lowrank.jl")
include("projectionmodels.jl")
include("datamatrix.jl")
include("subset_expression.jl")

include("adjacency_matrices.jl")

include("barnes_hut.jl")
include("force_layout.jl")
include("embed.jl")

include("h5ad.jl")

include("mannwhitney.jl")

include("filter.jl")
include("load.jl")
include("transform.jl")
include("design.jl")
include("normalize.jl")
include("reduce.jl")
include("annotate.jl")
include("statistical_tests.jl")
include("counts_fraction.jl")
include("counts_sum.jl")
include("pseudobulk.jl")

include("local_outlier_factor.jl")

include("precompile.jl")

@static if !isdefined(Base, :get_extension)
	function __init__()
		@require UMAP="c4f8c510-2410-5be4-91d7-4fbaeb39457e" include("../ext/SingleCellProjectionsUMAPExt.jl")
		@require TSne="24678dba-d5e9-5843-a4c6-250288b04835" include("../ext/SingleCellProjectionsTSneExt.jl")
		@require PrincipalMomentAnalysis="6a3ba550-3b7f-11e9-2734-d9178ad1e8db" include("../ext/SingleCellProjectionsPrincipalMomentAnalysisExt.jl")
		@require StableRNGs="860ef19b-820b-49d6-a774-d7a799459cd3" include("../ext/SingleCellProjectionsStableRNGsExt.jl")
		@require Muon="446846d7-b4ce-489d-bf74-72da18fe3629" include("../ext/SingleCellProjectionsMuonExt.jl")
	end
end

end
