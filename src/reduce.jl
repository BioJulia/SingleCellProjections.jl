function actual_nsv_pr(::Action, data, nsv)
	# NB: nsv is determined completely by the base case, so we do not project
	P = nvar_job(data)
	N = nobs_job(data)
	create_job(min, nsv, P, N; __version=v"0.1.0")
end
actual_nsv_job(data, nsv) = create_job(Projectable(actual_nsv_pr), data, nsv)


function implicitsvd_impl(matrix; kwargs...)
	progress = ProgressBar(styled"{blue:  ┌─}")
	F = SCPCore.implicitsvd(matrix; progress, tick=throw_if_cancelled, kwargs...)
	CompoundResult(; F.U, F.S, F.Vt)
end
function implicitsvd_job(matrix;
                          nsv,
                          seed,
                          subspacedims = 4nsv,
                          niter = 3,
                          stabilize_sign = true,
                          kwargs...)
	create_job(implicitsvd_impl, matrix; nsv, seed, subspacedims, niter, stabilize_sign, kwargs..., __version=v"0.1.1") # must be used with cached() to handle the CompoundResult
end

svd_projected_svt_job(U, X) =
	cached(create_job(SCPCore.svd_projected_svt, U, X; __version=v"0.1.0"))

svd_project_mul_sinv_job(ΣVt, S) =
	cached(create_job(SCPCore.svd_project_mul_sinv, ΣVt, S; __version=v"0.1.0"))

assemble_svd(U, S, Vt) = create_job(LinearAlgebra.SVD, U, S, Vt; __version=v"0.1.0")


# helpers
_svd_U_job(svd_job) = cached(svd_job, "U")
_svd_S_job(svd_job) = cached(svd_job, "S")
_svd_Vt_job(svd_job) = cached(svd_job, "Vt")

function svd_pr(action::Action, matrix; kwargs...)
	# First SVD of unprojected
	svd_job = implicitsvd_job(matrix; kwargs...)
	U = _svd_U_job(svd_job) # unaffected by projection
	S = _svd_S_job(svd_job) # unaffected by projection
	if action isa Eval
		Vt = _svd_Vt_job(svd_job)
	else#if action isa Projection
		ΣVt = svd_projected_svt_job(U, action(matrix))
		Vt = svd_project_mul_sinv_job(ΣVt, S)
	end
	assemble_svd(U, S, Vt)
end

function svd(::Mat, data; nsv, kwargs...)
	nsv = fetched(actual_nsv_job(data, nsv))
	create_job(Projectable(svd_pr), get_matrix_job(data); nsv, kwargs...)
end
svd(f::Union{Var,Obs}, data; kwargs...) = get_job(f, data)


function Jobs.svd(matrix; nsv, seed=1234, kwargs...)
	create_job(DataMatrixFunction(svd), matrix; nsv, seed, kwargs...)
end






compute_components(S, Vt) = LinearAlgebra.Diagonal(S)*Vt
compute_components_job(S, Vt) = create_job(compute_components, S, Vt; __version=v"0.1.0")

function pca_pr(action::Action, matrix; kwargs...)
	# First SVD of unprojected
	svd_job = implicitsvd_job(matrix; kwargs...)
	S = _svd_S_job(svd_job) # unaffected by projection
	if action isa Eval
		Vt = _svd_Vt_job(svd_job)
		compute_components_job(S, Vt)
	else#if action isa Projection
		U = _svd_U_job(svd_job) # unaffected by projection
		svd_projected_svt_job(U, action(matrix))
	end
end

# This is needed to ensure nsv is fetched - also in the projection case.
pca_pre(::Preprocessing, matrix; kwargs...) =
	create_job(Projectable(pca_pr), matrix; kwargs...)

function pca(::Mat, data; nsv, kwargs...)
	nsv = fetched(actual_nsv_job(data, nsv))
	create_job(Preprocess(pca_pre), get_matrix_job(data); nsv, kwargs...)
end
function pca(::Var, data; nsv, kwargs...)
	nsv = fetched(actual_nsv_job(data, nsv))
	prefixed_ids_job("PC_id", "PC", nsv)
end
pca(::Obs, data; kwargs...) = get_job(Obs(), data)

function Jobs.pca(data; nsv, seed=1234, kwargs...)
	create_job(DataMatrixFunction(pca), data; nsv, seed, kwargs...)
end




function loadings_pr(::Action, matrix; kwargs...)
	# Loadings are not affected by projection
	svd_job = implicitsvd_job(matrix; kwargs...)
	_svd_U_job(svd_job)
end

function loadings(::Mat, data; nsv, kwargs...)
	nsv = fetched(actual_nsv_job(data, nsv))
	create_job(Projectable(loadings_pr), get_matrix_job(data); nsv, kwargs...)
end
loadings(::Var, data; kwargs...) = get_job(Var(), data)
function loadings(::Obs, data; nsv, kwargs...)
	nsv = fetched(actual_nsv_job(data, nsv))
	prefixed_ids_job("loadings_id", "loadings", nsv)
end

function Jobs.loadings(args...; nsv, seed=1234, kwargs...)
	create_job(DataMatrixFunction(loadings), args...; nsv, seed, kwargs...)
end






# embed_points(weighted_adj, matrix) = matrix*weighted_adj
# create_embed_points_job(weighted_adj, matrix) =
# 	cached(create_job(embed_points, weighted_adj, matrix; __version=v"0.1.0"))

function embed_points(f, base_data, base_reduced::AbstractMatrix{T}, data, indices) where T
	base_N = size(base_data,2)
	N = size(data,2)
	@assert size(base_data,1) == size(data,1)
	@assert size(base_data,2) == size(base_reduced,2)
	@assert N == size(indices,2)

	out = zeros(T, size(base_reduced,1), N)

	# for j in 1:N # TODO: Thread
	# tforeach(1:N) do j # Configure scheduler?
	tforeach(1:N; scheduler=:greedy, chunking=true, minchunksize=128) do j # TODO: Revisit parameters
		total_weight = 0.0
		for base_j in @view(indices[:,j])
			# d2 = sum(abs2, @view(data[:,j]) .- @view(base_data[:,base_j])) # allocates
			d2 = mapreduce((a,b)->abs2(a-b), +, @view(data[:,j]), @view(base_data[:,base_j])) # TODO: Maybe write optimized function for this with @inbounds and @simd?
			w = f(d2)
			out[:,j] .+= w.*@view(base_reduced[:,base_j])
			total_weight += w
		end

		out[:,j] .*= 1.0./total_weight
	end

	out
end
create_embed_points_job(f, base_data, base_reduced, data, indices) =
	cached(create_job(embed_points, f, base_data, base_reduced, data, indices; __version=v"0.2.0"))


function force_layout_impl(args...; kwargs...)
	progress = ProgressBar(styled"{blue:  ┌─}")
	# SCPCore.force_layout(args...; progress, kwargs...)
	SCPCore.force_layout(args...; progress, tick=throw_if_cancelled, kwargs...)
end

function force_layout(action::Action, matrix;
                      k = nothing,
                      k_fraction = nothing,
                      make_symmetric=true,
                      ndim = 3,
                      niter = 100,
                      link_distance = 40,
                      link_strength = 0.05,
                      charge = 40,
                      charge_min_distance=1,
                      theta = 0.9,
                      center_strength=0.05,
                      velocity_decay=0.9,
                      initialAlpha = 1.0, finalAlpha = 1e-3,
                      initialScale = 10,
                      seed = 1234,
                      k_projection = 10, # TODO: support _fraction here as well.
                      min_dist2_projection = 1e-12,
                     )

	# First force layout of unprojected
	knn_indices = cached(find_nearest_neighbors_job(matrix; k, k_fraction))
	adj_job = adjacency_matrix_job(knn_indices; make_symmetric)

	fl_job = cached(create_job(force_layout_impl, adj_job;
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
		return fl_job
	else#if actions isa Projection
		knn_indices_p = cached(find_nearest_neighbors_job(matrix, action(matrix); k=k_projection))
		return create_embed_points_job(InvMax(min_dist2_projection), matrix, fl_job, action(matrix), knn_indices_p)
	end
end




function force_layout(::Mat, data; kwargs...)
	matrix_job = get_matrix_job(data)
	create_job(Projectable(force_layout), matrix_job; kwargs...)
end
force_layout(::Obs, data; kwargs...) = get_job(Obs(), data)
force_layout(::Var, data; ndim, kwargs...) = prefixed_ids_job("id", "Force Layout Dim ", ndim)

function Jobs.force_layout(args...; ndim=3, kwargs...)
	create_job(DataMatrixFunction(force_layout), args...; ndim, kwargs...)
end
