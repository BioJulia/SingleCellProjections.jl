value_vector_model_spec(annot; kwargs...) =
	cached(create_spec(SCPCore.value_vector_model, annot; kwargs..., __version=v"0.1.0"))


function value_vector(action::Action, annot; kwargs...)
	model = value_vector_model_spec(annot; kwargs...)
	cached(create_spec(SCPCore.value_vector, model, action(annot); __version=v"0.1.0"))
end
value_vector_spec(annot; kwargs...) =
	create_spec(Projectable(value_vector), annot; kwargs...)



covariate_model(action::Action, value_vector; kwargs...) =
	cached(create_spec(SCPCore.covariate_model, value_vector; kwargs..., __version=v"0.1.0"))
covariate_model_spec(value_vector; kwargs...) =
	create_spec(Projectable(covariate_model), value_vector; kwargs...)

covariate(action::Action, model, value_vector) =
	create_spec(SCPCore.covariate_matrix, prefetched(model), action(value_vector); __version=v"0.1.0") # Should we use cached()?
covariate_spec(model, value_vector) =
	create_spec(Projectable(covariate), model, value_vector)




covariate_scale(action::Action, model) = create_spec(SCPCore.covariate_scale, action(model); __version=v"0.1.0")
covariate_scale_spec(model) = create_spec(Projectable(covariate_scale), model)






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
	DataFrame(covariate=cov_names; copycols=false)
end


covariate_names2(action::Action, names, models) =
	create_spec(covariate_names_impl2, names, models; __version=v"0.1.0")
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
	cached(create_spec(SCPCore.value_vector_model, data, desc; kwargs..., __version=v"0.1.2"))

value_vector2(action::Action, model, data) =
	cached(create_spec(SCPCore.value_vector, model, action(data); __version=v"0.1.2"))
value_vector_spec2(model, data) =
	create_spec(Projectable(value_vector2), model, data)

# Special case for intercept, just the number of rows
_value_vector_data_spec(obs, ::String, ::SCPCore.InterceptCovariateDesc) =
	table_nrow_spec(obs)

# Column in obs
_value_vector_data_spec(obs, column::String, ::Any) = column_data_spec(obs, column)

# External annotation (DataFrame or spec)
function _value_vector_data_spec(obs, annot, ::Any)
	ids_a = id_column_spec(obs)
	ids_b = id_column_spec(annot)
	ind_spec = indexin_spec(ids_a, ids_b; not_found=:nothing)
	v = value_column_data_spec(annot)
	getindex_or_missing_spec(v, ind_spec) # The values of the annotation `k`, reordered to match the order in df.
end







build_designmatrix(::Mat, ::Any, covariates, ::Any) = create_hcat_spec(covariates...)
build_designmatrix(::Obs, ::Any, ::Any, covariate_names) = covariate_names
build_designmatrix(::Var, data, ::Any, ::Any) = get_obs_spec(data) # Yes this is correct. (See note below regarding transposing)

build_designmatrix_spec(data, covariates::Vector, covariate_names) =
	create_spec(DataMatrixFunction(build_designmatrix), data, covariates, covariate_names)


function setup_covariate_descriptions(args...; center)
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

	covariate_descriptions, center
end


function covariate_stages(obs, covariate_descriptions; center, kwargs...)
	vv_data_specs = _value_vector_data_spec.(Ref(obs), first.(covariate_descriptions), last.(covariate_descriptions))
	vv_model_specs = prefetched.(value_vector_model_spec2.(vv_data_specs, last.(covariate_descriptions)))
	vv_specs = prefetched.(value_vector_spec2.(vv_model_specs, vv_data_specs))

	covariate_model_specs = covariate_model_spec.(vv_specs; center, kwargs...)
	covariate_specs = covariate_spec.(covariate_model_specs, vv_specs)

	base_name_specs = _covariate_basename.(covariate_descriptions)
	covariate_names = covariate_names_spec2(base_name_specs, vv_model_specs)

	(; covariate_descriptions, vv_data_specs, vv_model_specs, vv_specs, covariate_model_specs, covariate_specs, covariate_names)
end



function designmatrix(::Preprocessing, data, args...; center, kwargs...)
	obs = get_obs_spec(data)
	covariate_descriptions, center = setup_covariate_descriptions(args...; center)
	(; covariate_specs, covariate_names) = covariate_stages(obs, covariate_descriptions; center, kwargs...)
	build_designmatrix_spec(data, covariate_specs, covariate_names)
end

# Creates a DataMatrix with obs as var and covariates as obs
# TODO: Consider transposing
# data::DataMatrix, args are covariates (names, two-column DataFrames with IDs+Values or covariate descriptions), center::Bool
designmatrix_spec(data, args...; center=true, kwargs...) =
	create_spec(Preprocess(designmatrix), data, args...; center, kwargs...)
function Jobs.designmatrix(data, args...; kwargs...)
	Job(designmatrix_spec(data, args...; kwargs...))
end
