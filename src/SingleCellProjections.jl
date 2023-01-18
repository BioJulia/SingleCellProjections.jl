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
	project,
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
	normalization_model,
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
	differentialexpression

using LinearAlgebra
import LinearAlgebra: svd

using SparseArrays
using ThreadedSparseArrays
using Statistics

using HDF5, H5Zblosc
using DataFrames

using Missings

using Random

using NearestNeighbors
using StaticArrays

using Distributions

using Requires

import SCTransform: SCTransform, scparams, sctransform
using SingleCell10x


include("MatrixExpressions/MatrixExpressions.jl")
using .MatrixExpressions


include("utils.jl")
include("table_utils.jl")


include("bilinear.jl")
include("sctransformsparse.jl")


include("implicitsvd.jl")


include("barnes_hut.jl")
include("force_layout.jl")
include("embed.jl")

include("h5ad.jl")

include("lowrank.jl")
include("projectionmodels.jl")
include("datamatrix.jl")
include("subset_expression.jl")
include("filter.jl")
include("load.jl")
include("transform.jl")
include("normalize.jl")
include("reduce.jl")
include("annotate.jl")
include("counts_fraction.jl")

include("differentialexpression.jl")

include("precompile.jl")

function __init__()
    @require UMAP="c4f8c510-2410-5be4-91d7-4fbaeb39457e" include("umap_glue.jl")
    @require TSne="24678dba-d5e9-5843-a4c6-250288b04835" include("tsne_glue.jl")
    @require PrincipalMomentAnalysis="6a3ba550-3b7f-11e9-2734-d9178ad1e8db" include("pma_glue.jl")
end

end
