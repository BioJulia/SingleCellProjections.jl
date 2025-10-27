struct Projectable{F} <: AbstractPreprocess{F}
	f::F
end

struct DataMatrixFunction{F} <: AbstractPreprocess{F}
	f::F
end

struct TableFunction{F} <: AbstractPreprocess{F}
	f::F
end

struct SetupTable{F} <: AbstractPreprocess{F}
	f::F
end
Base.show(io::IO, p::SetupTable{F}) where F = print(io, "SetupTable(", p.f, ')')
