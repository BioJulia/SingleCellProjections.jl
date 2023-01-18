struct UMAPModel <: ProjectionModel
	m::UMAP.UMAP_
	var_match::DataFrame
	obs::Symbol
end

projection_isequal(m1::UMAPModel, m2::UMAPModel) = m1.m === m2.m && m1.var_match == m2.var_match

update_model(m::UMAPModel; obs=m.obs, kwargs...) = (UMAPModel(m.m, m.var_match, obs), kwargs)


function UMAP.umap(data::DataMatrix, args...; obs=:copy, kwargs...)
	model = UMAPModel(UMAP.UMAP_(obs_coordinates(data), args...; kwargs...), select(data.var,data.var_id_cols), obs)
	update_matrix(data, model.m.embedding, model; var="UMAP", model.obs)
end

function project_impl(data::DataMatrix, model::UMAPModel; verbose=true)
	@assert table_cols_equal(data.var, model.var_match) "UMAP projection expects model and data variables to be identical."

	matrix = UMAP.transform(model.m, obs_coordinates(data))
	update_matrix(data, matrix, model; var="UMAP", model.obs)
end

# - show -
function Base.show(io::IO, ::MIME"text/plain", model::UMAPModel)
	print(io, "UMAP(n_components=", size(model.m.embedding,1), ')')
end
