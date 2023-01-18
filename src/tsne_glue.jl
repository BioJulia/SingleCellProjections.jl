function TSne.tsne(data::DataMatrix, args...; k=10, obs=:copy, kwargs...)
	t = permutedims(TSne.tsne(obs_coordinates(data)', args...; kwargs...))
	model = NearestNeighborModel("tsne",data.matrix, t, select(data.var,data.var_id_cols), k, "t-SNE", obs)
	update_matrix(data, t, model; model.var, model.obs)
end
