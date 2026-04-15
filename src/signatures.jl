# TODO: Figure out a nice interface for getting loadings as well
function signature_pre(::Preprocessing, data, var_filter, out_col_name; kwargs...)
	data = Jobs.filter_var(var_filter, data; project_var_ids=:yes) # We want exactly the same variables when projecting. Otherwise we cannot trust the projected signature.
	data = Jobs.normalize_matrix(data) # center
	reduced = Jobs.pca(data; nsv=1, subspacedims=10, kwargs...) # Hmm. Do we want to stabilize the sign differently?
	pc1 = get_matrix_row_spec(Jobs.get_matrix(reduced), 1)
	obs_ids = id_column_spec(get_obs_spec(reduced))
	table_hcat_spec(obs_ids, create_table_spec(out_col_name=>pc1))
end

signature_spec(data, var_filter, out_col_name; kwargs...) =
	create_spec(Preprocess(signature_pre), data, var_filter, out_col_name; kwargs...)
function Jobs.signature(data, var_filter, out_col_name; kwargs...)
	signature_spec(data, var_filter, out_col_name)
end
