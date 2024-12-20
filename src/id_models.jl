struct ObsIdModel <: ProjectionModel end
struct VarIdModel <: ProjectionModel end

project2(::ObsIdModel, data::DataMatrix) = data.obs[!,1]
project2(::VarIdModel, data::DataMatrix) = data.var[!,1]


struct DataMatrixModel <: ProjectionModel end

project2(::DataMatrixModel, data, obs, var) = DataMatrix(data, obs, var)
