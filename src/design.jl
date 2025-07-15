value_vector_model_spec(annot; kwargs...) =
	create_spec(SCPCore.value_vector_model, annot; __use_cache=true, kwargs..., __version=v"0.1.0")


function value_vector(action::Action, annot; kwargs...)
	model = value_vector_model_spec(annot; kwargs...)
	create_spec(SCPCore.value_vector_project, model, action(annot); __use_cache=true, __version=v"0.1.0")
end
value_vector_spec(annot; kwargs...) =
	create_spec(Projectable(value_vector), annot; kwargs...)


function covariate(action::Action, args...; kwargs...)
	model = create_spec(SCPCore.covariate_model, args...; __use_cache=true, kwargs..., __version=v"0.1.0")
	create_spec(SCPCore.covariate_project, model, action(args)...; __use_cache=false, __version=v"0.1.0") # What should __use_cache be?
end
covariate_spec(args...; kwargs...) =
	create_spec(Projectable(covariate), args...; kwargs...)



# TODO: Move these into SingleCellProjections.jl?
function _add_covariate_names!(out, name, model::SCPCore.CategoricalValueVectorModel)
	for c in model.categories
		push!(out, string(name, '_', c))
	end
end
_add_covariate_names!(out, name, ::SCPCore.NumericalValueVectorModel) = push!(out, name)
function covariate_names_impl(v::Vector{<:Pair{String,<:Any}}; center::Bool)
	cov_names = String[]
	center && push!(cov_names, "Intercept")
	for (name,model) in v
		_add_covariate_names!(cov_names, name, model)
	end
	@assert allunique(cov_names)
	DataFrame(covariate=cov_names)
end


covariate_names(action::Action, args...; center) =
	create_spec(covariate_names_impl, action(args)...; center, __use_cache=false, __version=v"0.1.0")
covariate_names_spec(args...; center) =
	create_spec(Projectable(covariate_names), args...; center)



function design(f::Union{Mat,Obs}, data, args...; center)
	# TODO: early out if only centering or doing nothing at all

	obs = get_obs_spec(data)
	annotation_specs = [create_extract_annotation_spec(obs, a) for a in args]

	if f isa Mat
		value_vector_specs = prefetched.(value_vector_spec.(annotation_specs))
		covariate_specs = covariate_spec.(value_vector_specs; center)

		if center
			nrows_spec = prefetched(annotation_nrows_spec(obs))
			intercept_spec = covariate_spec(nrows_spec; center)
			return create_hcat_spec(intercept_spec, covariate_specs...)
		else
			return create_hcat_spec(covariate_specs...)
		end
	else #if f isa Obs
		value_vector_model_specs = value_vector_model_spec.(annotation_specs)
		return covariate_names_spec(args .=> value_vector_model_specs; center)
	end
end

function design(::Var, data, args...; center)
	# Yes this is correct. (See note below regarding transposing)
	get_obs_spec(data)
end



# WIP
# Creates a DataMatrix with obs as var and covariates as obs
# TODO: Consider transposing
# data::DataMatrix, args are covariates (names), center::Bool
designmatrix_spec(data, args...; center=true, kwargs...) =
	create_spec(DataMatrixFunc(design), data, args...; center, kwargs...)
function Jobs.designmatrix(data, args...; kwargs...)
	Job(designmatrix_spec(data, args...; kwargs...))
end
