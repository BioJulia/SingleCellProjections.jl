# TODO: Move some of these specs to other files?

function neighbor_distances_impl(args...; kwargs...)
	progress = ProgressBar(styled"{blue:  ┌─}")
	SCPCore.neighbor_distances(args...; progress, kwargs...)
end


neighbor_distances_job(indices, X, DX2, args...) =
	cached(create_job(neighbor_distances_impl, indices, X, DX2, args...; __version=v"0.1.0"))


local_reachability_density_job(indices, dists, kdists) =
	cached(create_job(SCPCore.local_reachability_density, indices, dists, kdists; __version=v"0.1.0"))


local_outlier_factor_impl_job(indices, lrd, args...) =
	cached(create_job(SCPCore.local_outlier_factor, indices, lrd, args...; __version=v"0.1.0"))



function local_outlier_factor(action::Action, mat, full_mat; k)
	knn_indices = find_nearest_neighbors_job(mat; k)
	sum_squared = col_sum_squared_job(full_mat)
	full_dists = neighbor_distances_job(knn_indices, full_mat, sum_squared)

	kdists = cached(apply_job(maximum, full_dists; dims=1))
	lrd = local_reachability_density_job(knn_indices, full_dists, kdists)

	if action isa Eval
		local_outlier_factor_impl_job(knn_indices, lrd)
	else#if actions isa Projection

		mat2 = action(mat)
		full_mat2 = action(full_mat)

		knn_indices2 = find_nearest_neighbors_job(mat, mat2; k)
		sum_squared2 = col_sum_squared_job(full_mat2)
		full_dists2 = neighbor_distances_job(knn_indices2, full_mat, sum_squared, full_mat2, sum_squared2)
		lrd2 = local_reachability_density_job(knn_indices2, full_dists2, kdists) # NB: kdists are from the base case

		local_outlier_factor_impl_job(knn_indices2, lrd, lrd2)
	end
end




function local_outlier_factor_pre(::Preprocessing, data, full; k, col::String)
	mat = get_matrix_job(data)
	full_mat = get_matrix_job(full)

	obs_ids = id_column_job(get_obs_job(data)) # TODO: check that this is equal to `id_column_job(get_obs_job(full))`

	lof_job = create_job(Projectable(local_outlier_factor), mat, full_mat; k)
	table_hcat_job(obs_ids, create_table_job(col=>lof_job))
end


local_outlier_factor_job(data, full; k=10, col="LOF") =
	create_job(Preprocess(local_outlier_factor_pre), data, full; k, col)
"""
    Jobs.local_outlier_factor(data, full; k=10, col="LOF") -> Job

Compute the [Local Outlier Factor](https://en.wikipedia.org/wiki/Local_outlier_factor)
for each observation in `data` relative to the full dataset `full`, using `k` nearest
neighbors. Returns a table with IDs and LOF scores in a column named `col`.

When projecting, only neighbors in the base dataset are considered.
"""
function Jobs.local_outlier_factor(data, full; kwargs...)
	local_outlier_factor_job(data, full; kwargs...)
end
