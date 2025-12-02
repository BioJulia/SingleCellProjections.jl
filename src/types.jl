struct Projectable{F} <: AbstractPreprocess{F}
	f::F
end

struct ProjectOnto{F} <: AbstractPreprocess{F}
	f::F
end


struct DataMatrixFunction{F} <: AbstractPreprocess{F}
	f::F
end
