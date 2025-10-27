struct Projectable{F} <: AbstractPreprocess{F}
	f::F
end

struct DataMatrixFunction{F} <: AbstractPreprocess{F}
	f::F
end

struct TableFunction{F} <: AbstractPreprocess{F}
	f::F
end

struct ColNamesTableFunction{F} <: AbstractPreprocess{F} # TODO: Find a better name
	f::F
end
