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

MatFunction(f::F) where F = MatFunction{F}(f)
VarFunction(f::F) where F = VarFunction{F}(f)
ObsFunction(f::F) where F = ObsFunction{F}(f)


# The alias will not print unless exported, so we do this:
Base.show(io::IO, p::MatFunction{F}) where {F} = print(io, "MatFunction(", p.f, ')')
Base.show(io::IO, p::VarFunction{F}) where {F} = print(io, "VarFunction(", p.f, ')')
Base.show(io::IO, p::ObsFunction{F}) where {F} = print(io, "ObsFunction(", p.f, ')')


function StableHashTraits.transformer(::Type{S}) where S<:DataMatrixFieldFunction{T} where {T}
	# include name of T to distinguish between different AbstractPreprocess instances with the same function
	StableHashTraits.Transformer(x->(nameof(S), nameof(T), x.f))
end


# TODO: Use these for normal show as well? Or perhaps MatFunction/VarFunction/ObsFunction.
ReproducibleJobs.styled_function_name(x::MatFunction) = ReproducibleJobs.styled_function_name(x.f) * styled" {bright_black:(get_matrix)}"
ReproducibleJobs.styled_function_name(x::VarFunction) = ReproducibleJobs.styled_function_name(x.f) * styled" {bright_black:(get_var)}"
ReproducibleJobs.styled_function_name(x::ObsFunction) = ReproducibleJobs.styled_function_name(x.f) * styled" {bright_black:(get_obs)}"
