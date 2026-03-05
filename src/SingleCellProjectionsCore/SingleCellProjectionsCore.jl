module SingleCellProjectionsCore

# export
# 	DataMatrix,
# 	ProjectionModel, # deprecated
# 	FilterModel, # deprecated
# 	LogTransformModel, # deprecated
# 	LogTransformModel2, # deprecated
# 	TFIDFTransformModel, # deprecated
# 	SCTransformModel, # deprecated
# 	DesignMatrixModel, # deprecated
# 	NormalizationModel, # deprecated
# 	SVDModel, # deprecated
# 	NearestNeighborModel, # deprecated
# 	ObsAnnotationModel, # deprecated
# 	VarCountsFractionModel, # deprecated
# 	VarCountsSumModel, # deprecated
# 	PseudoBulkModel, # deprecated
# 	project,
# 	get_matrix,
# 	get_var,
# 	get_obs,
# 	get_var_ids,
# 	get_obs_ids,
# 	set_var_id_col!,
# 	set_obs_id_col!,
# 	var_coordinates,
# 	obs_coordinates,
# 	load10x,
# 	loadh5ad,
# 	load_counts,
# 	merge_counts,
# 	filter_matrix,
# 	filter_var,
# 	filter_obs,
# 	covariate,
# 	auto_covariate,
# 	intercept_covariate,
# 	categorical_covariate,
# 	numerical_covariate,
# 	twogroup_covariate,
# 	center_matrix, # TEMP
# 	designmatrix,
# 	normalize_matrix,
# 	svd,
# 	update_matrix,
# 	logtransform,
# 	sctransform,
# 	tf_idf_transform,
# 	knn_adjacency_matrix,
# 	knn_adjacency_matrix2,
# 	force_layout,
# 	var_to_obs!,
# 	var_to_obs,
# 	var_to_obs_table,
# 	var_counts_fraction!,
# 	var_counts_fraction,
# 	var_counts_sum!,
# 	var_counts_sum,
# 	pseudobulk,
# 	local_outlier_factor!,
# 	local_outlier_factor,
# 	local_outlier_factor_table,
# 	local_outlier_factor_projection!,
# 	local_outlier_factor_projection,
# 	local_outlier_factor_projection_table,
# 	ftest!,
# 	ftest,
# 	ftest_table,
# 	ttest!,
# 	ttest,
# 	ttest_table,
# 	mannwhitney!,
# 	mannwhitney,
# 	mannwhitney_table,
# 	create_datamatrix,
# 	create_var,
# 	create_obs

using LinearAlgebra

using SparseArrays
using Statistics

using ReadOnlyArrays

using HDF5, H5Zblosc
using DataFrames

using Missings

using Random
using StableRNGs

using NearestNeighbors
using StaticArrays

using Distributions

import SCTransform: SCTransform, scparams, sctransform
using SingleCell10x

using ChunkSplitters
using OhMyThreads


include("../MatrixExpressions/MatrixExpressions.jl")
using .MatrixExpressions


# # This symbol is only defined on Julia versions that support extensions
# isdefined(Base, :get_extension) || using Requires


# TODO: Is this still needed?
const Index = Union{AbstractVector, Colon}


include("random.jl")
include("utils.jl")
include("table_utils.jl")
include("threaded_sparse_row_map.jl")

include("bilinear.jl")
include("sctransformsparse.jl")

include("implicitsvd.jl")

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
include("statistical_tests.jl")
include("counts_fraction.jl")
include("counts_sum.jl")

include("annotation_transfer.jl")

include("local_outlier_factor.jl")

# include("precompile.jl")

end
