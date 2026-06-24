module TSneExt

using ReproducibleJobs
using ReproducibleJobs: create_job, cached
using SingleCellProjections
using SingleCellProjections: DataMatrixFunction, Projectable, Action, Eval, Projection, Mat, Var, Obs, get_matrix_job, get_job, prefixed_ids_job, find_nearest_neighbors_job, create_embed_points_job, InvMax
import TSne

# """
# 	tsne(data::DataMatrix, args...; k=10, obs=:copy, kwargs...)

# Create a t-SNE embedding of `data`.
# Usually, `data` is a DataMatrix after reduction to `10-100` dimensions by `svd`.

# * `k` - The number of neighbors to use when projecting onto a t-SNE model. (Not used in the t-SNE computation, only during projection.)
# * `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

# The other `args...` and `kwargs...` are forwarded to `TSne.tsne`. See `TSne` documentation for more details.

# See also: [`TSne.tsne`](https://github.com/lejon/TSne.jl)
# """
# function TSne.tsne(data::DataMatrix, args...; k=10, obs=:copy, kwargs...)
# 	t = permutedims(TSne.tsne(obs_coordinates(data)', args...; kwargs...))
# 	model = NearestNeighborModel("tsne", data, t; k, var="t-SNE", obs)
# 	update_matrix(data, t, model; model.var, model.obs)
# end


function tsne_impl(matrix; ndim, max_iter, perplexity, kwargs...)
	permutedims(TSne.tsne(parent(matrix)', ndim, 0, max_iter, perplexity; pca_init=true, progress=false, kwargs...))
end

function tsne(action::Action, matrix;
              ndim,
              max_iter = 1000,
              perplexity = 30,
              k_projection = 10,
              min_dist2_projection = 1e-12,
              kwargs...
             )
	# t-SNE of unprojected
	tsne_job = cached(create_job(tsne_impl, matrix; ndim, max_iter, perplexity, __version=v"0.1.0"))

	if action isa Eval
		return tsne_job
	else#if actions isa Projection
		knn_indices_p = cached(find_nearest_neighbors_job(matrix, action(matrix); k=k_projection))
		return create_embed_points_job(InvMax(min_dist2_projection), matrix, tsne_job, action(matrix), knn_indices_p)
	end
end




function tsne(::Mat, data; kwargs...)
	matrix_job = get_matrix_job(data)
	create_job(Projectable(tsne), matrix_job; kwargs...)
end
tsne(::Obs, data; kwargs...) = get_job(Obs(), data)
tsne(::Var, data; ndim, kwargs...) = prefixed_ids_job("id", "t-SNE ", ndim)

"""
    Jobs.tsne(data; ndim=3, kwargs...) -> Job

Compute a t-SNE embedding of `data` with `ndim` dimensions. Returns a `DataMatrix` with
t-SNE dimensions as variables. Requires the `TSne` package to be loaded.

See also `Jobs.force_layout`, `Jobs.umap`.
"""
function Jobs.tsne(args...; ndim=3, kwargs...)
	create_job(DataMatrixFunction(tsne), args...; ndim, kwargs...)
end



end
