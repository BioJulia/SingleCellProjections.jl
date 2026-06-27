"""
    SCP.nvar(data) -> Job

Return a `Job` for the number of variables (rows) in `data`.

See also [`nobs`](@ref).
"""
nvar(data) = Impl.nvar_job(data)

"""
    SCP.nobs(data) -> Job

Return a `Job` for the number of observations (columns) in `data`.

See also [`nvar`](@ref).
"""
nobs(data) = Impl.nobs_job(data)
