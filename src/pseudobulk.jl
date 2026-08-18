"""
    SCP.pseudobulk(data, obs_covariate1, obs_covariates...; kwargs...) -> Job

Aggregate single-cell data into pseudobulk by grouping observations according to the
specified covariates. Returns a `DataMatrix` where each column is a pseudobulk sample.

(TODO: Add example.)

See also [`population_matrix`](@ref).
"""
function pseudobulk(data, obs_covariate1, obs_covariates...; kwargs...)
	check_kwargs(kwargs, :new_var_covariates, :delim, :var_id_colname, :obs_id_colname)
	create_job(Preprocess(Impl.pseudobulk), data, obs_covariate1, obs_covariates...; kwargs...)
end


"""
    SCP.population_matrix(obs, obs_covariate1, obs_covariates...; new_var_covariates, kwargs...) -> Job

Create a matrix where each entry is the fraction of cells belonging to each combination of
`new_var_covariates` within each group defined by the observation covariates. The observation
covariates define the columns (samples/groups) and `new_var_covariates` define the rows
(e.g. cell type proportions per sample).

(TODO: Add an example.)

See also [`pseudobulk`](@ref).
"""
function population_matrix(obs, obs_covariate1, obs_covariates...; new_var_covariates, kwargs...)
	check_kwargs(kwargs, :delim, :var_id_colname, :obs_id_colname)
	create_job(Preprocess(Impl.population_matrix), obs, obs_covariate1, obs_covariates...; new_var_covariates, kwargs...)
end
