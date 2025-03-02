abstract type ProjectionModel end
Base.Broadcast.broadcastable(m::ProjectionModel) = Ref(m) # treat as scalar for broadcasting




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
