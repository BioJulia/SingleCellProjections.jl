"""
    SCP.signature(data, var_filter, out_col_name; loadings=false, kwargs...) -> Job

Compute a gene signature score for each observation by filtering to genes matching
`var_filter`, normalizing, and extracting the first principal component. Returns a table
with IDs and the signature scores in a column named `out_col_name`.

(TODO: Example.)

See also [`pca`](@ref), [`loadings`](@ref).
"""
function signature(data, var_filter, out_col_name; kwargs...)
	Impl.signature_job(data, var_filter, out_col_name; kwargs...)
end
