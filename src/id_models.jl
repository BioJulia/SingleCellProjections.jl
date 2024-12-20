struct ObsIdModel end
struct VarIdModel end

project2(data::DataMatrix, ::ObsIdModel) = data.obs[!,1]
project2(data::DataMatrix, ::VarIdModel) = data.var[!,1]
