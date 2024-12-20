struct ObsIdModel end
struct VarIdModel end

project2(::ObsIdModel, data::DataMatrix) = data.obs[!,1]
project2(::VarIdModel, data::DataMatrix) = data.var[!,1]
