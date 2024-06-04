module SingleCellProjectionsUMAPExt

using SingleCellProjections
using DataFrames
isdefined(Base, :get_extension) ? (using UMAP) : (using ..UMAP)

struct UMAPModel <: ProjectionModel
	m::UMAP.UMAP_
	var_match::DataFrame
	obs::Symbol
end

SingleCellProjections.projection_isequal(m1::UMAPModel, m2::UMAPModel) = m1.m == m2.m && m1.var_match == m2.var_match

SingleCellProjections.update_model(m::UMAPModel; obs=m.obs, kwargs...) = (UMAPModel(m.m, m.var_match, obs), kwargs)


"""
	umap(data::DataMatrix, args...; obs=:copy, kwargs...)

Create a UMAP embedding of `data`.
Usually, `data` is a DataMatrix after reduction to `10-100` dimensions by `svd`.

* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

The other `args...` and `kwargs...` are forwarded to `UMAP.umap`. See `UMAP` documentation for more details.

See also: [`UMAP.umap`](https://github.com/dillondaudert/UMAP.jl)
"""
function UMAP.umap(data::DataMatrix, args...; obs=:copy, kwargs...)
	model = UMAPModel(UMAP.UMAP_(obs_coordinates(data), args...; kwargs...), select(data.var,1), obs)
	update_matrix(data, model.m.embedding, model; var="UMAP", model.obs)
end

function SingleCellProjections.project_impl(data::DataMatrix, model::UMAPModel; verbose=true)
	@assert SingleCellProjections.table_cols_equal(data.var, model.var_match) "UMAP projection expects model and data variables to be identical."

	matrix = UMAP.transform(model.m, obs_coordinates(data))
	update_matrix(data, matrix, model; var="UMAP", model.obs)
end

# - show -
function Base.show(io::IO, ::MIME"text/plain", model::UMAPModel)
	print(io, "UMAPModel(n_components=", size(model.m.embedding,1), ')')
end


end
