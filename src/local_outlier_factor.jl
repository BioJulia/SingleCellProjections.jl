

col_sum_squared_spec(X) =
	create_spec(SCPCore.col_sum_squared, X; __version=v"0.0.1")


nearest_neighbor_distances_spec(indices, X, DX2) =
	create_spec(SCPCore.nearest_neighbor_distances, indices, X, DX2; __version=v"0.0.1")


local_reachability_density_spec(indices, dists, kdists) =
	create_spec(SCPCore.local_reachability_density, indices, dists, kdists; __version=v"0.0.1")


local_outlier_factor_impl_spec(indices, lrd) =
	create_spec(SCPCore.local_outlier_factor, indices, lrd; __version=v"0.0.1")



function local_outlier_factor(action::Action, mat, full_mat; k)
	knn_indices = find_nearest_neighbors_spec(mat; k)
	sum_squared = col_sum_squared_spec(full_mat)
	full_dists = nearest_neighbor_distances_spec(knn_indices, full_mat, sum_squared)

	kdists = apply_spec(maximum, full_dists; dims=1)
	lrd = local_reachability_density_spec(knn_indices, full_dists, kdists)

	if action isa Eval
		local_outlier_factor_impl_spec(knn_indices, lrd)
	else#if actions isa Projection
		error("Not yet implemented.")
	end
end




function local_outlier_factor_pre(::Preprocessing, data, full; k, col::String)
	mat = get_matrix_spec(data)
	full_mat = get_matrix_spec(full)

	obs_ids = id_column_spec(get_obs_spec(data)) # TODO: check that this is equal to `id_column_spec(get_obs_spec(full))`

	lof_spec = create_spec(Projectable(local_outlier_factor), mat, full_mat; k)
	table_hcat_spec(obs_ids, create_table_spec(col=>lof_spec))
end


local_outlier_factor_spec(data, full; k=10, col="LOF") =
	create_spec(Preprocess(local_outlier_factor_pre), data, full; k, col)
function Jobs.local_outlier_factor(data, full; kwargs...)
	local_outlier_factor_spec(data, full; kwargs...)
end
