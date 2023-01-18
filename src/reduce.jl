struct SVDModel <: ProjectionModel
	F::SVD
	var_match::DataFrame
	var::Symbol
	obs::Symbol
end

# NB: Require Factorizations to be ===. This is faster and more reasonable.
# Because of numerical issues, we should never expect Factorizations to be equal if they are not identical.
projection_isequal(m1::SVDModel, m2::SVDModel) = m1.F === m2.F && m1.var_match == m2.var_match


update_model(m::SVDModel; var=m.var, obs=m.obs, kwargs...) = (SVDModel(m.F, m.var_match, var, obs), kwargs)



function LinearAlgebra.svd(data::DataMatrix; nsv=3, var=:copy, obs=:copy, kwargs...)
	F = implicitsvd(data.matrix; nsv=nsv, kwargs...)
	model = SVDModel(F, select(data.var,data.var_id_cols), var, obs)
	update_matrix(data, F, model; model.var, model.obs)
end


function project_impl(data::DataMatrix, model::SVDModel; verbose=true)
	@assert table_cols_equal(data.var, model.var_match) "SVD projection expects model and data variables to be identical."

	F = model.F
	X = data.matrix

	V = X'F.U # TODO: compute F.U'X instead to get Vt directly
	V ./= F.S'
	matrix = SVD(F.U,F.S,Matrix(V'))
	update_matrix(data, matrix, model; model.obs, model.var)
end


struct NearestNeighborModel <: ProjectionModel
	name::String
	pre::Matrix{Float64}
	post::Matrix{Float64}
	var_match::DataFrame
	k::Int
	var::String
	obs::Symbol
end
NearestNeighborModel(name, pre::DataMatrix, post; k, var, obs) =
	NearestNeighborModel(name, obs_coordinates(pre), post, select(pre.var,pre.var_id_cols), k, var, obs)
NearestNeighborModel(name, pre::DataMatrix, post::DataMatrix; kwargs...) =
	NearestNeighborModel(name, pre, obs_coordinates(post); kwargs...)

function projection_isequal(m1::NearestNeighborModel, m2::NearestNeighborModel)
	# NB: === for factorizations/matrices
	m1.name == m2.name && m1.pre === m2.pre && m1.post === m2.post &&
	m1.var_match == m2.var_match && m1.k == m2.k
end

update_model(m::NearestNeighborModel; k=m.k, var=m.var, obs=m.obs, kwargs...) =
	(NearestNeighborModel(m.name, m.pre, m.post, m.var_match, k, var, obs), kwargs)


function project_impl(data::DataMatrix, model::NearestNeighborModel; adj_out=nothing, verbose=true)
	@assert table_cols_equal(data.var, model.var_match) "Nearest Neighbor projection expects model and data variables to be identical."

	adj, matrix = embed_points(model.pre, model.post, obs_coordinates(data); model.k)
	adj_out !== nothing && (adj_out[] = adj)

	update_matrix(data, matrix, model; model.var, model.obs)
end


function force_layout(data::DataMatrix; ndim=3, k=nothing, adj=nothing, kprojection=10, obs=:copy, adj_out=nothing, kwargs...)
	@assert isnothing(k) != isnothing(adj) "Must specify exactly one of k and adj."
	if k !== nothing
		adj = knn_adjacency_matrix(obs_coordinates(data); k)
	end
	adj_out !== nothing && (adj_out[] = adj)

	fl = force_layout(adj; ndim, kwargs...)
	model = NearestNeighborModel("force_layout", data, fl; k=kprojection, var="Force Layout Dim ", obs)
	update_matrix(data, fl, model; model.var, model.obs)
end


# - show -
function Base.show(io::IO, ::MIME"text/plain", model::SVDModel)
	print(io, "SVDModel(nsv=", innersize(model.F), ')')
end
function Base.show(io::IO, ::MIME"text/plain", model::NearestNeighborModel)
	print(io, "NearestNeighborModel(base=\"", model.name, "\", k=", model.k, ')')
end
