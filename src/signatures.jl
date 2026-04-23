function signature_pre(::Preprocessing, data, var_filter, out_col_name; loadings=false, extra_cols=(), kwargs...)
	data = Jobs.filter_var(var_filter, data; project_var_ids=:yes) # We want exactly the same variables when projecting. Otherwise we cannot trust the projected signature.
	data = Jobs.normalize_matrix(data) # center

	svd_kwargs = (; nsv=1, subspacedims=10, kwargs...) # Hmm. Do we want to stabilize the sign differently?

	if loadings
		reduced = Jobs.loadings(data; svd_kwargs...)
		pc1 = get_matrix_col_spec(Jobs.get_matrix(reduced), 1)
		annot = get_var_spec(reduced)
	else
		reduced = Jobs.pca(data; svd_kwargs...)
		pc1 = get_matrix_row_spec(Jobs.get_matrix(reduced), 1)
		annot = get_obs_spec(reduced)
	end

	extra_cols isa Union{Symbol,<:AbstractString} && (extra_cols = (extra_cols,)) # for splatting convenience
	table = get_columns_spec(annot, fetched(get_id_colname_spec(annot)), extra_cols...)
	table_hcat_spec(table, create_table_spec(out_col_name=>pc1))
end

signature_spec(data, var_filter, out_col_name; kwargs...) =
	create_spec(Preprocess(signature_pre), data, var_filter, out_col_name; kwargs...)
function Jobs.signature(data, var_filter, out_col_name; kwargs...)
	signature_spec(data, var_filter, out_col_name; kwargs...)
end
