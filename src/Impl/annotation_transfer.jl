function transfer_annotation_impl(f, base_data, base_annot, data, indices, ::SCPCore.CategoricalCovariateDesc)
	transferred, score = SCPCore.transfer_categorical_annotation(f, base_data, base_annot, data, indices)
	CompoundResult(; transferred, score)
end
transfer_annotation_impl_job(f, base_data, base_annot, data, indices, desc) =
	create_job(transfer_annotation_impl, f, base_data, base_annot, data, indices, desc; __version=v"1.0.0")





function update_name(old_name; new_name=nothing, new_suffix=nothing)
	if new_name !== nothing
		new_name
	elseif new_suffix !== nothing
		string(old_name, new_suffix)
	else
		old_name
	end
end
update_name_job(old_name; kwargs...) =
	create_job(update_name, old_name; kwargs..., __version=v"1.0.0")


function transfer_annotation(::Preprocessing, base, new, covariate; k, weight_fun=InvMax(1e-12), kwargs...)
	# TODO: Check that var agree
	base_mat = SCP.get_matrix(base)
	new_mat = SCP.get_matrix(new)

	# TODO: reimplement without actually constructing dists as an intermediate representation

	# knn = find_nearest_neighbors_job(base_mat, new_mat; k)
	# # Unwrap knn_job CompoundResult
	# indices = cached(knn, "indices")
	# dists = cached(knn, "distances")
	# wadj = weighted_adjacency_matrix_job(weight_fun, indices, dists; NX=SCP.nobs(base))


	knn_indices = find_nearest_neighbors_job(base_mat, new_mat; k)

	obs = SCP.get_obs(base)
	annot, desc = setup_covariate_description(obs, covariate)
	annot_name = fetched(_extract_name(annot))
	annot_data = _extract_data_job(obs, annot)

	# t = transfer_annotation_impl_job(wadj, annot_data, desc)
	t = transfer_annotation_impl_job(weight_fun, base_mat, annot_data, new_mat, knn_indices, desc)
	transferred = cached(t, "transferred")
	score = cached(t, "score")

	# make into table
	new_obs_ids = SCP.id_column(SCP.get_obs(new))
	transferred_name = fetched(update_name_job(annot_name; kwargs...))
	score_name = fetched(update_name_job(transferred_name; new_suffix="_score"))
	SCP.table_hcat(new_obs_ids, SCP.create_table(transferred_name=>transferred, score_name=>score))
end
