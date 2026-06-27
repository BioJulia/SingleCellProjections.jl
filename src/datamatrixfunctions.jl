"""
    SCP.get_matrix(data) -> Job

Extract the matrix component from a `DataMatrix` `Job`.

See also [`get_var`](@ref), [`get_obs`](@ref).
"""
get_matrix(x) = Impl.get_matrix_job(x)

"""
    SCP.get_var(data) -> Job

Extract the variable annotation table from a `DataMatrix` `Job`.

See also [`get_matrix`](@ref), [`get_obs`](@ref).
"""
get_var(x) = Impl.get_var_job(x)

"""
    SCP.get_obs(data) -> Job

Extract the observation annotation table from a `DataMatrix` `Job`.

See also [`get_matrix`](@ref), [`get_var`](@ref).
"""
get_obs(x) = Impl.get_obs_job(x)
