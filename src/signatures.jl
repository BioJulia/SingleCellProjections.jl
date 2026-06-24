function signature_pre(::Preprocessing, data, var_filter, out_col_name; loadings=false, extra_cols=(), kwargs...)
	data = Jobs.filter_var(var_filter, data; project_var_ids=:yes) # We want exactly the same variables when projecting. Otherwise we cannot trust the projected signature.
	data = Jobs.normalize_matrix(data) # center

	svd_kwargs = (; nsv=1, subspacedims=10, kwargs...) # Hmm. Do we want to stabilize the sign differently?

	if loadings
		reduced = Jobs.loadings(data; svd_kwargs...)
		pc1 = get_matrix_col_job(Jobs.get_matrix(reduced), 1)
		annot = get_var_job(reduced)
	else
		reduced = Jobs.pca(data; svd_kwargs...)
		pc1 = get_matrix_row_job(Jobs.get_matrix(reduced), 1)
		annot = get_obs_job(reduced)
	end

	extra_cols isa Union{Symbol,<:AbstractString} && (extra_cols = (extra_cols,)) # for splatting convenience
	table = get_columns_job(annot, fetched(get_id_colname_job(annot)), extra_cols...)
	table_hcat_job(table, create_table_job(out_col_name=>pc1))
end

signature_job(data, var_filter, out_col_name; kwargs...) =
	create_job(Preprocess(signature_pre), data, var_filter, out_col_name; kwargs...)
"""
    Jobs.signature(data, var_filter, out_col_name; loadings=false, kwargs...) -> Job

Compute a gene signature score for each observation by filtering to genes matching
`var_filter`, normalizing, and extracting the first principal component. Returns a table
with IDs and the signature scores in a column named `out_col_name`.

(TODO: Example.)

See also [`Jobs.pca`](@ref), [`Jobs.loadings`](@ref).
"""
function Jobs.signature(data, var_filter, out_col_name; kwargs...)
	signature_job(data, var_filter, out_col_name; kwargs...)
end
