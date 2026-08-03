"""
    SCP.subset_matrix(data, var_ids, obs_ids) -> Job

Subset `data` to keep only variables and observations with IDs present in `var_ids` and `obs_ids`.

See also [`subset_var`](@ref), [`subset_obs`](@ref).
"""
subset_matrix(data, var_ids, obs_ids) =
	create_job(Preprocess(Impl.subset_matrix), data; var_ids, obs_ids)
"""
    SCP.subset_var(data, var_ids) -> Job

Subset `data` to keep only variables with IDs present in `var_ids`.

See also [`subset_obs`](@ref), [`subset_matrix`](@ref), [`filter_var`](@ref).
"""
subset_var(data, var_ids) =
	create_job(Preprocess(Impl.subset_matrix), data; var_ids)
"""
    SCP.subset_obs(data, obs_ids) -> Job

Subset `data` to keep only observations with IDs present in `obs_ids`.

See also [`subset_var`](@ref), [`subset_matrix`](@ref), [`filter_obs`](@ref).
"""
subset_obs(data, obs_ids) =
	create_job(Preprocess(Impl.subset_matrix), data; obs_ids)


"""
    SCP.filter_matrix(fvar, fobs, data; kwargs...) -> Job

Filter both variables and observations simultaneously.

See also [`filter_var`](@ref), [`filter_obs`](@ref).
"""
filter_matrix(fvar, fobs, data; kwargs...) =
	create_job(Preprocess(Impl.filter_matrix), data; kwargs..., fvar, fobs)
"""
    SCP.filter_var(fvar, data; kwargs...) -> Job

Filter variables by the predicate `fvar`. `fvar` can be:
- An integer range or vector of indices (e.g. `1:100`).
- A `Pair` of column name and predicate (e.g. `"name" => >("D")`).
- An annotation table `Job` with a predicate (e.g. `SCP.relative_std(data) => >=(0.1)`).

See also [`filter_obs`](@ref), [`filter_matrix`](@ref), [`subset_var`](@ref).
"""
filter_var(fvar, data; kwargs...) =
	create_job(Preprocess(Impl.filter_matrix), data; kwargs..., fvar)
"""
    SCP.filter_obs(fobs, data; kwargs...) -> Job

Filter observations by the predicate `fobs`. `fobs` can be:
- An integer range or vector of indices.
- A `Pair` of column name and predicate (e.g. `"celltype" => isequal("Monocyte")`).
- An annotation table `Job` with a predicate.

(TODO: Add an example with predicate. Cannot use the relative_std one.)

See also [`filter_var`](@ref), [`filter_matrix`](@ref), [`subset_obs`](@ref).
"""
filter_obs(fobs, data; kwargs...) =
	create_job(Preprocess(Impl.filter_matrix), data; kwargs..., fobs)
