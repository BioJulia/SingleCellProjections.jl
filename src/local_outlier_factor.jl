# TODO: Move some of these specs to other files?

function neighbor_distances_impl(args...; kwargs...)
	progress = ProgressBar(styled"{blue:  ┌─}")
	SCPCore.neighbor_distances(args...; progress, kwargs...)
end


neighbor_distances_spec(indices, X, DX2, args...) =
	cached(create_spec(neighbor_distances_impl, indices, X, DX2, args...; __version=v"0.1.0"))


local_reachability_density_spec(indices, dists, kdists) =
	cached(create_spec(SCPCore.local_reachability_density, indices, dists, kdists; __version=v"0.1.0"))


local_outlier_factor_impl_spec(indices, lrd, args...) =
	cached(create_spec(SCPCore.local_outlier_factor, indices, lrd, args...; __version=v"0.1.0"))



function local_outlier_factor(action::Action, mat, full_mat; k)
	knn_indices = find_nearest_neighbors_spec(mat; k)
	sum_squared = col_sum_squared_spec(full_mat)
	full_dists = neighbor_distances_spec(knn_indices, full_mat, sum_squared)

	kdists = cached(apply_spec(maximum, full_dists; dims=1))
	lrd = local_reachability_density_spec(knn_indices, full_dists, kdists)

	if action isa Eval
		local_outlier_factor_impl_spec(knn_indices, lrd)
	else#if actions isa Projection

		mat2 = action(mat)
		full_mat2 = action(full_mat)

		knn_indices2 = find_nearest_neighbors_spec(mat, mat2; k)
		sum_squared2 = col_sum_squared_spec(full_mat2)
		full_dists2 = neighbor_distances_spec(knn_indices2, full_mat, sum_squared, full_mat2, sum_squared2)
		lrd2 = local_reachability_density_spec(knn_indices2, full_dists2, kdists) # NB: kdists are from the base case

		local_outlier_factor_impl_spec(knn_indices2, lrd, lrd2)
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
