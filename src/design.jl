value_vector_model_spec(annot; kwargs...) =
	create_spec(SCPCore.value_vector_model, annot; __use_cache=true, kwargs..., __version=v"0.1.0")


function value_vector(action::Action, annot; kwargs...)
	model = value_vector_model_spec(annot; kwargs...)
	create_spec(SCPCore.value_vector_project, model, action(annot); __use_cache=true, __version=v"0.1.0")
end
value_vector_spec(annot; kwargs...) =
	create_spec(Projectable(value_vector), annot; kwargs...)


function covariate(action::Action, args...; kwargs...)
	model = prefetched(create_spec(SCPCore.covariate_model, args...; __use_cache=true, kwargs..., __version=v"0.1.0"))
	create_spec(SCPCore.covariate_project, model, action(args)...; __use_cache=false, __version=v"0.1.0") # What should __use_cache be?
end
covariate_spec(args...; kwargs...) =
	create_spec(Projectable(covariate), args...; kwargs...)



# TODO: We need to handle the case when "name" isn't a column name but a two-column dataframe with IDs + named colug
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
	# TODO: early out if only centering or doing nothing at all?

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
# designmatrix_spec(data, args...; center=true, kwargs...) =
# 	create_spec(DataMatrixFunction(design), data, args...; center, kwargs...)
# function Jobs.designmatrix(data, args...; kwargs...)
# 	Job(designmatrix_spec(data, args...; kwargs...))
# end




# --- Attempt at improving structure -------------------



# TODO: Move these into SingleCellProjections.jl?
function _add_covariate_names2!(out, name, model::SCPCore.CategoricalValueVectorModel)
	for c in model.categories
		push!(out, string(name, '_', c))
	end
end
_add_covariate_names2!(out, name, ::Any) = push!(out, name)

function covariate_names_impl2(names::Vector, models::Vector)
	@assert length(names)==length(models)
	cov_names = String[]
	for (name,model) in zip(names, models)
		_add_covariate_names2!(cov_names, name, model)
	end
	@assert allunique(cov_names)
	DataFrame(covariate=cov_names)
end


covariate_names2(action::Action, names, models) =
	create_spec(covariate_names_impl2, names, models; __use_cache=false, __version=v"0.1.0")
covariate_names_spec2(names, models) =
	create_spec(Projectable(covariate_names2), names, models)





function _covariate_basename((column, desc)::Pair)
	if desc === SCPCore.intercept_covariate()
		@assert column == "Intercept" # Allow other names?
		column
	elseif column isa Union{String,Symbol}
		column
	else
		annotation_name_spec(column)
	end
end




value_vector_model_spec2(data, desc::SCPCore.InterceptCovariateDesc; kwargs...) =
	SCPCore.InterceptValueVectorModel(; kwargs...)
value_vector_model_spec2(data, desc; kwargs...) =
	create_spec(SCPCore.value_vector_model, data, desc; __use_cache=true, kwargs..., __version=v"0.1.2")

value_vector2(action::Action, model, data; kwargs...) =
	create_spec(SCPCore.value_vector_project, model, action(data); __use_cache=true, __version=v"0.1.2")
value_vector_spec2(model, data) =
	create_spec(Projectable(value_vector2), model, data)

_value_vector_data_spec(obs, ::Any, ::SCPCore.InterceptCovariateDesc; kwargs...) =
	annotation_nrows_spec(obs) # Special case for intercept, just the number of rows
_value_vector_data_spec(obs, column, ::Any; kwargs...) =
	create_extract_annotation_spec(obs, column) # the column values






function build_designmatrix(::Mat, data, ::Any, value_vector_specs, ::Any; center, kwargs...)
	covariate_specs = covariate_spec.(value_vector_specs; center)
	create_hcat_spec(covariate_specs...)
end
build_designmatrix(::Obs, data, names, ::Any, models; kwargs...) = covariate_names_spec2(names, models)
build_designmatrix(::Var, data, ::Any, ::Any, ::Any; kwargs...) = get_obs_spec(data) # Yes this is correct. (See note below regarding transposing)

build_designmatrix_spec(data, names::Vector, value_vector_specs::Vector, value_vector_model_specs::Vector; kwargs...) =
	create_spec(DataMatrixFunction(build_designmatrix), data, names, value_vector_specs, value_vector_model_specs; kwargs...)



function designmatrix(data, args...; center, kwargs...)
	obs = get_obs_spec(data)

	# Automatically center if there is an intercept covariate
	center = center || any(a->a === SCPCore.intercept_covariate() || (a isa Pair && a.second === SCPCore.intercept_covariate()), args)

	# Gather covariates, ensuring at most one Intercept
	covariate_descriptions = Pair{Any,Any}[]
	center && push!(covariate_descriptions, "Intercept"=>SCPCore.intercept_covariate())
	for a in args
		if a isa Pair
			a.second === SCPCore.intercept_covariate() && continue # We have already handled the intercept
		else
			a === SCPCore.intercept_covariate() && continue # We have already handled the intercept
			a = a => SCPCore.auto_covariate()
		end
		push!(covariate_descriptions, a)
	end

	base_name_specs = _covariate_basename.(covariate_descriptions)
	vv_data_specs = _value_vector_data_spec.(Ref(obs), first.(covariate_descriptions), last.(covariate_descriptions))
	vv_model_specs = prefetched.(value_vector_model_spec2.(vv_data_specs, last.(covariate_descriptions)))
	vv_specs = prefetched.(value_vector_spec2.(vv_model_specs, vv_data_specs))
	build_designmatrix_spec(data, base_name_specs, vv_specs, vv_model_specs; center, kwargs...)
end

# Creates a DataMatrix with obs as var and covariates as obs
# TODO: Consider transposing
# data::DataMatrix, args are covariates (names, two-column DataFrames with IDs+Values or covariate descriptions), center::Bool
designmatrix_spec(data, args...; center=true, kwargs...) =
	create_spec(Preprocess(designmatrix), data, args...; center, kwargs...)
function Jobs.designmatrix(data, args...; kwargs...)
	Job(designmatrix_spec(data, args...; kwargs...))
end
