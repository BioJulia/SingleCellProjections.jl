module TSneExt

using ReproducibleJobs
using ReproducibleJobs: create_spec, cached
using SingleCellProjections
using SingleCellProjections: DataMatrixFunction, Projectable, Action, Eval, Projection, Mat, Var, Obs, get_matrix_spec, get_spec, prefixed_ids_spec, find_nearest_neighbors_spec, create_embed_points_spec, InvMax
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
	tsne_spec = cached(create_spec(tsne_impl, matrix; ndim, max_iter, perplexity, __version=v"0.1.0"))

	if action isa Eval
		return tsne_spec
	else#if actions isa Projection
		knn_indices_p = cached(find_nearest_neighbors_spec(matrix, action(matrix); k=k_projection))
		return create_embed_points_spec(InvMax(min_dist2_projection), matrix, tsne_spec, action(matrix), knn_indices_p)
	end
end




function tsne(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(tsne), matrix_spec; kwargs...)
end
tsne(::Obs, data; kwargs...) = get_spec(Obs(), data)
tsne(::Var, data; ndim, kwargs...) = prefixed_ids_spec("id", "t-SNE ", ndim)

function Jobs.tsne(args...; ndim=3, kwargs...)
	create_spec(DataMatrixFunction(tsne), args...; ndim, kwargs...)
end



end
