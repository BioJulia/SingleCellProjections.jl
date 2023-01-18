abstract type ProjectionModel end

projection_isequal(::ProjectionModel, ::ProjectionModel) = false
projection_isequal(model::ProjectionModel) = Base.Fix2(projection_isequal, model)


function _update_model(model::ProjectionModel; kwargs...)
	isempty(kwargs) && return (model,kwargs)
	update_model(model; kwargs...)
end


# Just a fallback if we forget to define show for a model
Base.show(io::IO, ::MIME"text/plain", model::T) where T<:ProjectionModel =
	print(io, replace(string(nameof(T)),"Model"=>""))

function Base.show(io::IO, model::T) where T<:ProjectionModel
	if get(io,:compact,false)
		print(io, replace(string(nameof(T)),"Model"=>""))
	else
		show(io, MIME"text/plain"(), model)
	end
end
