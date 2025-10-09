struct Projectable{F} <: AbstractPreprocess{F}
	f::F
end

struct DataMatrixFunc{F} <: AbstractPreprocess{F} # TODO: Can we find a better/shorter name?
	f::F
end
