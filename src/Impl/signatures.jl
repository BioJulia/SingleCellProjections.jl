function signature_pre(::Preprocessing, data, var_filter, out_col_name; loadings=false, extra_cols=(), kwargs...)
	data = SCP.filter_var(var_filter, data; project_var_ids=:yes) # We want exactly the same variables when projecting. Otherwise we cannot trust the projected signature.
	data = SCP.normalize_matrix(data) # center

	svd_kwargs = (; nsv=1, subspacedims=10, kwargs...) # Hmm. Do we want to stabilize the sign differently?

	if loadings
		reduced = SCP.loadings(data; svd_kwargs...)
		pc1 = get_matrix_col_job(SCP.get_matrix(reduced), 1)
		annot = SCP.get_var(reduced)
	else
		reduced = SCP.pca(data; svd_kwargs...)
		pc1 = get_matrix_row_job(SCP.get_matrix(reduced), 1)
		annot = SCP.get_obs(reduced)
	end

	extra_cols isa Union{Symbol,<:AbstractString} && (extra_cols = (extra_cols,)) # for splatting convenience
	table = SCP.get_columns(annot, fetched(SCP.get_id_colname(annot)), extra_cols...)
	SCP.table_hcat(table, SCP.create_table(out_col_name=>pc1))
end
