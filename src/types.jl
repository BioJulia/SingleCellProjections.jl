struct Projectable{F} <: AbstractPreprocess{F}
	f::F
end

struct ProjectOnto{F} <: AbstractPreprocess{F}
	f::F
end


struct DataMatrixFunction{F} <: AbstractPreprocess{F}
	f::F
end


abstract type DataMatrixField end
struct Mat <: DataMatrixField end
struct Var <: DataMatrixField end
struct Obs <: DataMatrixField end

struct DataMatrixFieldFunction{T,F} <: AbstractPreprocess{F}
	f::F
end

const MatFunction = DataMatrixFieldFunction{Mat}
const VarFunction = DataMatrixFieldFunction{Var}
const ObsFunction = DataMatrixFieldFunction{Obs}

# MatFunction(f::F) where F = MatFunction{F}(f)
# DEBUG
function MatFunction(f::F) where F
	# F <: DataMatrixFunction && error("hej")
	MatFunction{F}(f)
end
VarFunction(f::F) where F = VarFunction{F}(f)
ObsFunction(f::F) where F = ObsFunction{F}(f)


function StableHashTraits.transformer(::Type{S}) where S<:DataMatrixFieldFunction{T} where {T}
	# include name of T to distinguish between different AbstractPreprocess instances with the same function
	StableHashTraits.Transformer(x->(nameof(S), nameof(T), x.f))
end

