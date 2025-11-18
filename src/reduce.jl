function actual_nsv_pr(::Action, data, nsv)
	# NB: nsv is determined completely by the base case, so we do not project
	P = nvar_spec(data)
	N = nobs_spec(data)
	create_spec(min, nsv, P, N; __version=v"0.1.0")
end
actual_nsv_spec(data, nsv) = create_spec(Projectable(actual_nsv_pr), data, nsv)


function implicitsvd_impl(matrix; nsv, kwargs...) # nsv is a required kwarg
	F = SCPCore.implicitsvd(matrix; nsv, kwargs...)
	CompoundResult(; F.U, F.S, F.Vt)
end
implicitsvd_spec(matrix; kwargs...) =
	create_spec(implicitsvd_impl, matrix; kwargs..., __version=v"0.1.0") # must be used with cached() to handle the CompoundResult

svd_projected_vt_spec(U, S, X) =
	cached(create_spec(SCPCore.svd_projected_vt, U, S, X; __version=v"0.1.0"))

assemble_svd(U, S, Vt) = create_spec(LinearAlgebra.SVD, U, S, Vt; __version=v"0.1.0")


# helpers
_svd_U_spec(svd_spec) = cached(svd_spec, "U")
_svd_S_spec(svd_spec) = cached(svd_spec, "S")
_svd_Vt_spec(svd_spec) = cached(svd_spec, "Vt")

function svd_pr(action::Action, matrix; kwargs...)
	# First SVD of unprojected
	svd_spec = implicitsvd_spec(matrix; kwargs...)
	U = _svd_U_spec(svd_spec) # unaffected by projection
	S = _svd_S_spec(svd_spec) # unaffected by projection
	if action isa Eval
		Vt = _svd_Vt_spec(svd_spec)
	else#if action isa Projection
		Vt = svd_projected_vt_spec(U, S, action(matrix))
	end
	assemble_svd(U, S, Vt)
end

function svd(::Mat, data; nsv, kwargs...)
	nsv = prefetched(actual_nsv_spec(data, nsv))
	create_spec(Projectable(svd_pr), get_matrix_spec(data); nsv, kwargs...)
end
svd(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)


function Jobs.svd(matrix; nsv, seed=1234, kwargs...)
	Job(create_spec(DataMatrixFunction(svd), matrix; nsv, seed, kwargs...))
end






compute_components(S, Vt) = LinearAlgebra.Diagonal(S)*Vt
compute_components_spec(S, Vt) = create_spec(compute_components, S, Vt; __version=v"0.1.0")

function pca_pr(action::Action, matrix; kwargs...)
	# First SVD of unprojected
	svd_spec = implicitsvd_spec(matrix; kwargs...)
	S = _svd_S_spec(svd_spec) # unaffected by projection
	if action isa Eval
		Vt = _svd_Vt_spec(svd_spec)
	else#if action isa Projection
		U = _svd_U_spec(svd_spec) # unaffected by projection
		Vt = svd_projected_vt_spec(U, S, action(matrix))
	end

	compute_components_spec(S, Vt)
end


function pca(::Mat, data; nsv, kwargs...)
	nsv = prefetched(actual_nsv_spec(data, nsv))
	create_spec(Projectable(pca_pr), get_matrix_spec(data); nsv, kwargs...)
end
function pca(::Var, data; nsv, kwargs...)
	nsv = prefetched(actual_nsv_spec(data, nsv))
	prefixed_ids_spec("PC_id", "PC", nsv)
end
pca(::Obs, data; kwargs...) = get_spec(Obs(), data)

function Jobs.pca(data; nsv, seed=1234, kwargs...)
	Job(create_spec(DataMatrixFunction(pca), data; nsv, seed, kwargs...))
end




function loadings_pr(::Action, matrix; kwargs...)
	# Loadings are not affected by projection
	svd_spec = implicitsvd_spec(matrix; kwargs...)
	_svd_U_spec(svd_spec)
end

function loadings(::Mat, data; nsv, kwargs...)
	nsv = prefetched(actual_nsv_spec(data, nsv))
	create_spec(Projectable(loadings_pr), get_matrix_spec(data); nsv, kwargs...)
end
loadings(::Var, data; kwargs...) = get_spec(Var(), data)
function loadings(::Obs, data; nsv, kwargs...)
	nsv = prefetched(actual_nsv_spec(data, nsv))
	prefixed_ids_spec("loadings_id", "loadings", nsv)
end

function Jobs.loadings(args...; nsv, seed=1234, kwargs...)
	Job(create_spec(DataMatrixFunction(loadings), args...; nsv, seed, kwargs...))
end






knn_adjacency_matrix(action::Action, matrix; kwargs...) =
	cached(create_spec(SCPCore.knn_adjacency_matrix, action(matrix); kwargs..., __version=v"0.1.0"))
create_knn_adjacency_matrix_spec(matrix; kwargs...) =
	create_spec(Projectable(knn_adjacency_matrix), matrix; kwargs...)



# Whoa. Shorten this name.
function inv_dist_squared_adjacency_matrix2(X, Y; min_dist=1e-6, kwargs...)
	SCPCore.knn_adjacency_matrix2(X, Y; kwargs...) do x
		1.0 / max(min_dist, x)^2
	end
end
create_inv_dist_squared_adjacency_matrix2_spec(X, Y; kwargs...) =
	cached(create_spec(inv_dist_squared_adjacency_matrix2, X, Y; kwargs..., __version=v"0.1.0"))



embed_points(weighted_adj, matrix) = matrix*weighted_adj
create_embed_points_spec(weighted_adj, matrix) =
	cached(create_spec(embed_points, weighted_adj, matrix; __version=v"0.1.0"))



function force_layout(action::Action, matrix;
                      k=100,
                      make_symmetric=true,
	                  ndim=3,
	                  niter=100,
                      link_distance=4, link_strength=2,
                      charge=5, charge_min_distance=1, theta = 0.9,
                      center_strength=0.05,
                      velocity_decay=0.9,
                      initialAlpha = 1.0, finalAlpha = 1e-3,
                      initialScale = 10,
                      seed = 1234,
                      k_projection = 10,
                      min_dist_projection = 1e-6,
                     )
	# First force layout of unprojected
	adj_spec = create_knn_adjacency_matrix_spec(matrix; k, make_symmetric)
	fl_spec = cached(create_spec(SCPCore.force_layout, adj_spec;
	                             ndim,
	                             niter,
	                             link_distance, link_strength,
	                             charge, charge_min_distance, theta,
	                             center_strength,
	                             velocity_decay,
	                             initialAlpha, finalAlpha,
	                             initialScale,
	                             seed,
	                             __version=v"0.1.0",
	                     ))

	if action isa Eval
		return fl_spec
	else#if actions isa Projection
		weighted_adj_spec = create_inv_dist_squared_adjacency_matrix2_spec(matrix, action(matrix);
		                                                                   k=k_projection,
		                                                                   min_dist=min_dist_projection)
		return create_embed_points_spec(weighted_adj_spec, fl_spec)
	end
end




function force_layout(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(force_layout), matrix_spec; kwargs...)
end
force_layout(::Obs, data; kwargs...) = get_spec(Obs(), data)
force_layout(::Var, data; ndim, kwargs...) = prefixed_ids_spec("id", "Force Layout Dim ", ndim)

function Jobs.force_layout(args...; ndim=3, kwargs...)
	Job(create_spec(DataMatrixFunction(force_layout), args...; ndim, kwargs...))
end
