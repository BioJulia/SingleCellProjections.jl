"""
	tsne(data::DataMatrix, args...; k=10, obs=:copy, kwargs...)

Create a t-SNE embedding of `data`.
Usually, `data` is a DataMatrix after reduction to `10-100` dimensions by `svd`.

* `k` - The number of neighbors to use when projecting onto a t-SNE model. (Not used in the t-SNE computation, only during projection.)
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

The other `args...` and `kwargs...` are forwarded to `TSne.tsne`. See `TSne` documentation for more details.

See also: [`TSne.tsne`](@ref)
"""
function TSne.tsne(data::DataMatrix, args...; k=10, obs=:copy, kwargs...)
	t = permutedims(TSne.tsne(obs_coordinates(data)', args...; kwargs...))
	model = NearestNeighborModel("tsne", data, t; k, var="t-SNE", obs)
	update_matrix(data, t, model; model.var, model.obs)
end
