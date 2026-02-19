module SingleCellProjectionsUMAPExt

using ReproducibleJobs
using ReproducibleJobs: create_spec, cached
using SingleCellProjections
using SingleCellProjections: DataMatrixFunction, Projectable, Action, Eval, Projection, Mat, Var, Obs, get_matrix_spec, get_spec, prefixed_ids_spec
import .SingleCellProjectionsCore as SCPCore

using DataFrames


using UMAP: UMAP, UMAPResult

# struct UMAPModel <: SCPCore.ProjectionModel
# 	m::UMAP.UMAP_
# 	var_match::DataFrame
# 	obs::Symbol
# end

# SCPCore.projection_isequal(m1::UMAPModel, m2::UMAPModel) = m1.m == m2.m && m1.var_match == m2.var_match

# SCPCore.update_model(m::UMAPModel; obs=m.obs, kwargs...) = (UMAPModel(m.m, m.var_match, obs), kwargs)


# """
# 	umap(data::DataMatrix, args...; obs=:copy, kwargs...)

# Create a UMAP embedding of `data`.
# Usually, `data` is a DataMatrix after reduction to `10-100` dimensions by `svd`.

# * `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

# The other `args...` and `kwargs...` are forwarded to `UMAP.umap`. See `UMAP` documentation for more details.

# See also: [`UMAP.umap`](https://github.com/dillondaudert/UMAP.jl)
# """
# function UMAP.umap(data::DataMatrix, args...; obs=:copy, kwargs...)
# 	# model = UMAPModel(UMAP.UMAP_(SCPCore.obs_coordinates(data), args...; kwargs...), select(data.var,1), obs)
# 	model = UMAPModel(UMAP.UMAP_(data.matrix, args...; kwargs...), select(data.var,1), obs)
# 	SCPCore.update_matrix(data, model.m.embedding, model; var="UMAP", model.obs)
# end

# function SCPCore.project_impl(data::DataMatrix, model::UMAPModel; verbose=true, kwargs...)
# 	@assert SCPCore.table_cols_equal(data.var, model.var_match) "UMAP projection expects model and data variables to be identical."

# 	# matrix = UMAP.transform(model.m, SCPCore.obs_coordinates(data))
# 	matrix = UMAP.transform(model.m, data.matrix)
# 	SCPCore.update_matrix(data, matrix, model; var="UMAP", model.obs)
# end

# # - show -
# function Base.show(io::IO, ::MIME"text/plain", model::UMAPModel)
# 	print(io, "UMAPModel(n_components=", size(model.m.embedding,1), ')')
# end



# ReproducibleJobs version

umap_model(matrix; ndim::Int, kwargs...) = UMAP.fit(matrix, ndim; kwargs...)
# umap_embedding(result::UMAPResult) = result.embedding # TODO: Move back to this when UMAP 0.3 is released
umap_embedding(result::UMAPResult) = reduce(hcat, result.embedding) # Temporary fix for UMAP 0.2
umap_project(result::UMAPResult, matrix) = umap_embedding(UMAP.transform(result, matrix))



function umap_impl(action::Action, matrix; ndim, kwargs...)
	# First create UMAP model
	umap_model_spec = cached(create_spec(umap_model, matrix; ndim, kwargs..., __version=v"0.2.0"))

	if action isa Eval
		return create_spec(umap_embedding, umap_model_spec; __version=v"0.2.1")
	else# if action isa Projection
		return cached(create_spec(umap_project, umap_model_spec, action(matrix); __version=v"0.2.0"))
	end
end


function umap(::Mat, data; ndim, kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(umap_impl), matrix_spec; ndim, kwargs...)
end
umap(::Obs, data; ndim, kwargs...) = get_spec(Obs(), data)
umap(::Var, data; ndim, kwargs...) = prefixed_ids_spec("id", "UMAP ", ndim)


function Jobs.umap(data; ndim, kwargs...)
	Job(create_spec(DataMatrixFunction(umap), data; ndim, kwargs...))
end






end
