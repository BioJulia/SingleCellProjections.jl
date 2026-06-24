function subset_matrix(::Preprocessing, data; var_ids=nothing, obs_ids=nothing)
	var_ind_args = var_ids === nothing ? (;) : (; var_ind=indexin_job(var_ids, id_column_job(get_var_job(data)); not_found=:error))
	obs_ind_args = obs_ids === nothing ? (;) : (; obs_ind=indexin_job(obs_ids, id_column_job(get_obs_job(data)); not_found=:error))
	create_datamatrix_getindex_job(data; var_ind_args..., obs_ind_args...)
end


"""
    Jobs.subset_matrix(data, var_ids, obs_ids) -> Job

Subset `data` to keep only variables and observations with IDs present in `var_ids` and `obs_ids`.

See also `Jobs.subset_var`, `Jobs.subset_obs`.
"""
Jobs.subset_matrix(data, var_ids, obs_ids) =
	create_job(Preprocess(subset_matrix), data; var_ids, obs_ids)
"""
    Jobs.subset_var(data, var_ids) -> Job

Subset `data` to keep only variables with IDs present in `var_ids`.

See also `Jobs.subset_obs`, `Jobs.subset_matrix`, `Jobs.filter_var`.
"""
Jobs.subset_var(data, var_ids) =
	create_job(Preprocess(subset_matrix), data; var_ids)
"""
    Jobs.subset_obs(data, obs_ids) -> Job

Subset `data` to keep only observations with IDs present in `obs_ids`.

See also `Jobs.subset_var`, `Jobs.subset_matrix`, `Jobs.filter_obs`.
"""
Jobs.subset_obs(data, obs_ids) =
	create_job(Preprocess(subset_matrix), data; obs_ids)



function filter_matrix(::Preprocessing, data; fvar=nothing, fobs=nothing, project_var_ids=nothing, project_obs_ids=nothing)
	if fvar === nothing
		@assert project_var_ids === nothing
		fvar = Colon()
		project_var_ids = :no
	end
	if fobs === nothing
		@assert project_obs_ids === nothing
		fobs = Colon()
		project_obs_ids = :no
	end
	project_var_ids = @something(project_var_ids, :intersect)
	project_obs_ids = @something(project_obs_ids, :no)

	var_ind = create_find_matching_ind_job(fvar, get_var_job(data); project_ids=project_var_ids)
	obs_ind = create_find_matching_ind_job(fobs, get_obs_job(data); project_ids=project_obs_ids)

	create_datamatrix_getindex_job(data; var_ind, obs_ind)
end


"""
    Jobs.filter_matrix(fvar, fobs, data; kwargs...) -> Job

Filter both variables and observations simultaneously.

See also `Jobs.filter_var`, `Jobs.filter_obs`.
"""
Jobs.filter_matrix(fvar, fobs, data; kwargs...) =
	create_job(Preprocess(filter_matrix), data; kwargs..., fvar, fobs)
"""
    Jobs.filter_var(fvar, data; kwargs...) -> Job

Filter variables by the predicate `fvar`. `fvar` can be:
- An integer range or vector of indices (e.g. `1:100`).
- A `Pair` of column name and predicate (e.g. `"name" => >("D")`).
- An annotation table `Job` with a predicate (e.g. `Jobs.relative_std(data) => >=(0.1)`).

See also `Jobs.filter_obs`, `Jobs.filter_matrix`, `Jobs.subset_var`.
"""
Jobs.filter_var(fvar, data; kwargs...) =
	create_job(Preprocess(filter_matrix), data; kwargs..., fvar)
"""
    Jobs.filter_obs(fobs, data; kwargs...) -> Job

Filter observations by the predicate `fobs`. `fobs` can be:
- An integer range or vector of indices.
- A `Pair` of column name and predicate (e.g. `"celltype" => isequal("Monocyte")`).
- An annotation table `Job` with a predicate.

(TODO: Add an example with predicate. Cannot use the relative_std one.)

See also `Jobs.filter_var`, `Jobs.filter_matrix`, `Jobs.subset_obs`.
"""
Jobs.filter_obs(fobs, data; kwargs...) =
	create_job(Preprocess(filter_matrix), data; kwargs..., fobs)
