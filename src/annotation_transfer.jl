function transfer_annotation_impl(wadj, annot_data, ::SCPCore.CategoricalCovariateDesc)
	transferred, score = SCPCore.transfer_categorical_annotation(wadj, annot_data)
	CompoundResult(; transferred, score)
end
transfer_annotation_impl_spec(adj, annot_data, desc) =
	create_spec(transfer_annotation_impl, adj, annot_data, desc; __version=v"0.1.0")


function update_name(old_name; new_name=nothing, new_suffix=nothing)
	if new_name !== nothing
		new_name
	elseif new_suffix !== nothing
		string(old_name, new_suffix)
	else
		old_name
	end
end
update_name_spec(old_name; kwargs...) =
	create_spec(update_name, old_name; kwargs..., __version=v"0.1.0")


function transfer_annotation(::Preprocessing, base, new, covariate; k, weight_fun=InvDistSquared(1e-6), new_suffix="_transferred", kwargs...)
	# TODO: Check that var agree
	base_mat = get_matrix_spec(base)
	new_mat = get_matrix_spec(new)

	knn = find_nearest_neighbors_spec(base_mat, new_mat; k)
	# Unwrap knn_spec CompoundResult
	indices = cached(knn, "indices")
	dists = cached(knn, "distances")

	wadj = weighted_adjacency_matrix_spec(weight_fun, indices, dists; NX=nobs_spec(base))

	obs = get_obs_spec(base)
	annot, desc = setup_covariate_description(obs, covariate)
	annot_name = fetched(_extract_name(annot))
	annot_data = _extract_data_spec(obs, annot)

	t = transfer_annotation_impl_spec(wadj, annot_data, desc)
	transferred = cached(t, "transferred")
	score = cached(t, "score")


	# make into table
	new_obs_ids = id_column_spec(get_obs_spec(new))
	transferred_name = fetched(update_name_spec(annot_name; new_suffix, kwargs...))
	score_name = fetched(update_name_spec(transferred_name; new_suffix="_score"))
	table_hcat_spec(new_obs_ids, create_table_spec(transferred_name=>transferred, score_name=>score))
end

transfer_annotation_spec(base, new, covariate; k, kwargs...) =
	create_spec(Preprocess(transfer_annotation), base, new, covariate; k, kwargs...)

Jobs.transfer_annotation(base, new, covariate; k, kwargs...) =
	Job(transfer_annotation_spec(base, new, covariate; k, kwargs...))
