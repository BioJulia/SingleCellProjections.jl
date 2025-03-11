# SVD is an example where the model comes after the result. I.e. svd(data) => UΣVᵀ, but the model is just UΣ.
function svd(action::Action, matrix; kwargs...)
	# First SVD of unprojected
	svd_spec = create_spec(SCPCore.implicitsvd, matrix; use_cache=true, kwargs..., __version=v"0.1.0")

	if action isa Eval
		return svd_spec
	else# if action isa Projection
		return create_spec(SCPCore.svd_project, svd_spec, action(matrix); use_cache=true, __version=v"0.1.0")
	end
end


function svd(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(svd), matrix_spec; kwargs...)
end
svd(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)

function Jobs.svd(args...; seed=1234, kwargs...)
	Job(create_spec(DataMatrixFunc(svd), args...; seed, kwargs...))
end





# TODO: Get rid of this by passing the matrix object to SCPCore.knn_adjacency_matrix instead?
knn_adjacency_matrix_impl(matrix; kwargs...) =
	SCPCore.knn_adjacency_matrix(SCPCore.obs_coordinates(matrix); kwargs...)

knn_adjacency_matrix(action::Action, matrix; kwargs...) =
	create_spec(knn_adjacency_matrix_impl, action(matrix); kwargs..., use_cache=true, __version=v"0.1.0")
create_knn_adjacency_matrix_spec(matrix; kwargs...) =
	create_spec(Projectable(knn_adjacency_matrix), matrix; kwargs...)



# Whoa. Shorten this name.
function inv_dist_squared_adjacency_matrix2(X, Y; min_dist=1e-6, kwargs...)
	SCPCore.knn_adjacency_matrix2(SCPCore.obs_coordinates(X), SCPCore.obs_coordinates(Y); kwargs...) do x
		1.0 / max(min_dist, x)^2
	end
end
create_inv_dist_squared_adjacency_matrix2_spec(X, Y; kwargs...) =
	create_spec(inv_dist_squared_adjacency_matrix2, X, Y; kwargs..., use_cache=true, __version=v"0.1.0")



embed_points(weighted_adj, matrix) = matrix*weighted_adj
create_embed_points_spec(weighted_adj, matrix) =
	create_spec(embed_points, weighted_adj, matrix; use_cache=true, __version=v"0.1.0")



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
	fl_spec = create_spec(SCPCore.force_layout, adj_spec;
	                      ndim,
	                      niter,
	                      link_distance, link_strength,
	                      charge, charge_min_distance, theta,
	                      center_strength,
	                      velocity_decay,
	                      initialAlpha, finalAlpha,
	                      initialScale,
	                      seed,
	                      use_cache=true,
	                      __version=v"0.1.0",
	                     )

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
force_layout(::Var, data; ndim, kwargs...) = create_prefixed_ids_spec("id", "Force Layout Dim ", ndim)

function Jobs.force_layout(args...; ndim=3, kwargs...)
	Job(create_spec(DataMatrixFunc(force_layout), args...; ndim, kwargs...))
end
