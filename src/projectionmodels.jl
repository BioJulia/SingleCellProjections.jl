abstract type ProjectionModel end

# TODO: rename to project
function project2 end


# NB: StableHashTraits package extension ensures hashing is correct for F<:Function
struct StatelessModel{F} <: ProjectionModel
	f::F
end
StatelessModel(f::F) where {F<:Function} = StatelessModel{F}(f)
StatelessModel(::Type{T}) where T = StatelessModel{Type{T}}(T)

project2(m::StatelessModel{F}, args...; kwargs...) where F =
	m.f(args...; kwargs...)




# deprecated, will be removed
projection_isequal(::ProjectionModel, ::ProjectionModel) = false
projection_isequal(model::ProjectionModel) = Base.Fix2(projection_isequal, model)


# deprecated, will be removed
function _update_model(model::ProjectionModel; kwargs...)
	isempty(kwargs) && return (model,kwargs)
	update_model(model; kwargs...)
end

# - show -

# Just a fallback if we forget to define show for a model
Base.show(io::IO, ::MIME"text/plain", model::T) where T<:ProjectionModel =
	print(io, nameof(T))

function Base.show(io::IO, model::T) where T<:ProjectionModel
	show(io, MIME"text/plain"(), model)
end


Base.show(io::IO, ::MIME"text/plain", model::StatelessModel{F}) where F =
	print(io, "StatelessModel(", model.f, ")")
