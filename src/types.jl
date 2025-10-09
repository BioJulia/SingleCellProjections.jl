struct Projectable{F} <: AbstractPreprocess{F}
	f::F
end

struct DataMatrixFunction{F} <: AbstractPreprocess{F}
	f::F
end
