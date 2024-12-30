# struct ObsIdModel <: ProjectionModel end
# struct VarIdModel <: ProjectionModel end

# project2(::ObsIdModel, data::DataMatrix) = data.obs[!,1:1]
# project2(::VarIdModel, data::DataMatrix) = data.var[!,1:1]

# Use StatelessModel{get_var_ids}, StatelessModel{get_obs_ids} instead


# struct DataMatrixModel <: ProjectionModel end
# project2(::DataMatrixModel, data, var, obs) = DataMatrix(data, var, obs)

# Use StatelessModel{DataMatrix} instead
