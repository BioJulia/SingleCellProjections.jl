module SingleCellProjections

export
	SingleCellProjectionsCore, # TODO: make public instead
	DataMatrix,
	Jobs, # TODO: remove probably
	auto_covariate,
	intercept_covariate,
	categorical_covariate,
	numerical_covariate,
	twogroup_covariate

include("SingleCellProjectionsCore/SingleCellProjectionsCore.jl")

import .SingleCellProjectionsCore as SCPCore
using .SCPCore: DataMatrix, auto_covariate, intercept_covariate, categorical_covariate, numerical_covariate, twogroup_covariate

import SCTransform
import SingleCell10x
using DataFrames
import StableHashTraits
import LinearAlgebra

using ReproducibleJobs
using ReproducibleJobs: create_spec, cached, ReadOnly, SpecArgs, ChecksummedFilePath, Preprocessing, checksummedfilepath_job, ifelse_spec

using ReadOnlyArrays: ReadOnlyVector

ReproducibleJobs.unmanage_rec(x::DataMatrix) =
	DataMatrix(ReproducibleJobs.unmanage_rec.((x.matrix, x.var, x.obs))...)
ReproducibleJobs.unmanage_rec(x::SCPCore.AbstractValueVector) = x
ReproducibleJobs.unmanage_rec(x::SCPCore.AbstractValueVectorModel) = x
ReproducibleJobs.unmanage_rec(x::SCPCore.ProjectionModel) = x


# TODO: This is a temporary solution when refactoring, remove
module Jobs
	function project end
	function load_counts end
	function get_matrix end
	function get_var end
	function get_obs end
	function nvar end
	function nobs end
	function annotate end
	function annotate_var end
	function annotate_obs end
	function add_var_column end
	function add_obs_column end
	function var_counts_fraction end
	function var_counts_sum end
	function obs_counts_fraction end
	function obs_counts_sum end
	# function find_matching_ids end
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
	function negative_regression_matrix end
	function normalize_matrix end
	function svd end
	function pca end
	function loadings end
	function force_layout end
	function transpose end

	function ftest end
	function ttest end

	function create_table end
	function get_colnames end
	function get_id_colname end
	function get_value_colname end
	function get_columns end
	function id_column end
	function value_column end
	function annotation end
	function column_data end
	function id_column_data end
	function value_column_data end
	function table_nrow end
	function add_column end
	function table_leftjoin end
	function table_hcat end

	function load_csv end
	function umap end
	function tsne end
end



include("types.jl")
include("projectables.jl")
include("datamatrixfunctions.jl")
include("internal.jl")
include("tables.jl")
include("matrix_arithmetic.jl")
include("load.jl")
include("annotate.jl")
include("filter.jl")
include("transform.jl")
include("design.jl")
include("normalize.jl")
include("reduce.jl")
include("adjoint.jl")
include("statistical_tests.jl")


# include("precompile.jl")


end
