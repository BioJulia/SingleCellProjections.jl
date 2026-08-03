"""
    SCP.get_matrix(data) -> Job

Extract the matrix component from a `DataMatrix` `Job`.

See also [`get_var`](@ref), [`get_obs`](@ref).
"""
get_matrix(x) = create_job(Preprocess(Impl.get_matrix), x)

"""
    SCP.get_var(data) -> Job

Extract the variable annotation table from a `DataMatrix` `Job`.

See also [`get_matrix`](@ref), [`get_obs`](@ref).
"""
get_var(x) = create_job(Preprocess(Impl.get_var), x)

"""
    SCP.get_obs(data) -> Job

Extract the observation annotation table from a `DataMatrix` `Job`.

See also [`get_matrix`](@ref), [`get_var`](@ref).
"""
get_obs(x) = create_job(Preprocess(Impl.get_obs), x)
