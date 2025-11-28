# Yet another new attempt at a better interface
# Making it easier to handle with Specs

detect_covariate_desc(::AbstractVector{<:Union{Missing,Number}}) = SCPCore.numerical_covariate()
detect_covariate_desc(::AbstractVector) = SCPCore.categorical_covariate()

detect_covariate_desc_spec(values) = create_spec(detect_covariate_desc, values; __version=v"0.1.0")


# TODO: Move this to internal? It can be used in many places.
_extract_data_spec(obs, column::String) = column_data_spec(obs, column) # Column in obs
function _extract_data_spec(obs, annot) # External annotation (DataFrame or spec)
	ids_a = id_column_spec(obs)
	ids_b = id_column_spec(annot)
	ind_spec = indexin_spec(ids_a, ids_b; not_found=:nothing)
	v = value_column_data_spec(annot)
	getindex_or_missing_spec(v, ind_spec) # The values of the annotation, reordered to match the order in df.
end

_extract_name(column::String) = column
_extract_name(annot) = get_value_colname_spec(annot)



function _group_args(desc::SCPCore.TwoGroupCovariateDesc)
	desc.group_b === nothing ? (desc.group_a,) : (desc.group_a,desc.group_b)
end




function setup_covariate_descriptions(obs, args...)
	annots = []
	descs = []
	for a in args
		if a isa Pair
			@assert a.second isa SCPCore.AbstractCovariateDesc
			push!(annots, a.first)
			push!(descs, a.second)
		else
			push!(annots, a)
			push!(descs, fetched(detect_covariate_desc_spec(_extract_data_spec(obs, a))))
		end
	end
	annots, descs
end



function twogroup_values(v, group_a, group_b=nothing)
	if group_b === nothing
		ifelse.(isequal.(v, group_a), 1, 2)
	else
		groups = [group_a, group_b]
		ind = indexin(v, groups)
		if any(isnothing, ind)
			new_values = setdiff(unique(v), groups)
			error("Two-group vector has values not present in model. Got [", join(new_values, ','), "], but expected $(group_a) and $(group_b).")
		end
		convert(Vector{Int}, ind)
	end
end
twogroup_values_spec(v, group_a, args...) =
	create_spec(twogroup_values, v, group_a, args...; __version=v"0.1.0")




intercept_covariate_matrix(n) = trues(n, 1)



mean_and_scale_spec(v; center) = create_spec(SCPCore.mean_and_scale, v; center, __version=v"0.1.0")
mean_and_scale_spec(v, ::SCPCore.NumericalCovariateDesc; center) = mean_and_scale_spec(v; center)
mean_and_scale_spec(v, desc::SCPCore.TwoGroupCovariateDesc; center) =
	mean_and_scale_spec(twogroup_values_spec(v, _group_args(desc)...); center)



function numerical_covariate_matrix_impl(v, (m,s))
	N = length(v)
	x = reshape(v, N, 1) # So we return a N×1 matrix
	(x .- m)./s
end
numerical_covariate_matrix_impl_spec(v, ms) =
	create_spec(numerical_covariate_matrix_impl, v, ms; __version=v"0.1.0")
function numerical_covariate_matrix(action::Action, data; center)
	ms = fetched(mean_and_scale_spec(data; center)) # model - not affected by action
	numerical_covariate_matrix_impl_spec(action(data), ms)
end


categories_spec(data) = unique_spec(data)
function categorical_covariate_matrix_impl(ind, n_categories)
	N = length(ind)
	X = falses(N, n_categories)
	X[CartesianIndex.(1:N, ind)] .= true
	X
end
categorical_covariate_matrix_impl_spec(ind, n_categories) =
	create_spec(categorical_covariate_matrix_impl, ind, n_categories; __version=v"0.1.0")

_too_many_categories_error(len, max_categories) =
	throw(ArgumentError("$len categories in categorical variable, was this intended? Change max_categories (", max_categories, ") if you want to increase the number of allowed categories."))
_too_many_categories_error_spec(len, max_categories) = create_spec(_too_many_categories_error, len, max_categories)

function categorical_covariate_matrix(action::Action, data; max_categories=100)
	categories = categories_spec(data) # model - not affected by action

	# application of model
	data = action(data)
	ind = indexin_spec(data, categories)
	result = categorical_covariate_matrix_impl_spec(ind, fetched(length_spec(categories)))

	len = length_spec(categories)
	cond = apply_spec(<=(max_categories), len)
	ifelse_pr_spec(cond, result, _too_many_categories_error_spec(len, max_categories))
end



intercept_covariate_matrix_spec(n) = create_spec(intercept_covariate_matrix, n; __version=v"0.1.0")
numerical_covariate_matrix_spec(data; center) = create_spec(Projectable(numerical_covariate_matrix), data; center)
categorical_covariate_matrix_spec(data; kwargs...) = create_spec(Projectable(categorical_covariate_matrix), data; kwargs...)

twogroup_covariate_matrix(::Preprocessing, data, args...; center) =
	numerical_covariate_matrix_spec(twogroup_values_spec(data, args...); center)
twogroup_covariate_matrix_spec(data, groups...; center) =
	create_spec(Preprocess(twogroup_covariate_matrix), data, groups...; center)


covariate_matrix_spec(data, desc::SCPCore.NumericalCovariateDesc; center, kwargs...) =
	numerical_covariate_matrix_spec(data; center)
function covariate_matrix_spec(data, desc::SCPCore.CategoricalCovariateDesc; max_categories=nothing, kwargs...)
	kw = max_categories !== nothing ? (; max_categories) : (;)
	categorical_covariate_matrix_spec(data; kw...)
end
covariate_matrix_spec(data, desc::SCPCore.TwoGroupCovariateDesc; kwargs...) =
	twogroup_covariate_matrix_spec(data, _group_args(desc)...; kwargs...)


extract_covariate_names(::Preprocessing, ::Any, ::SCPCore.NumericalCovariateDesc, basename) = basename
function extract_covariate_names(::Preprocessing, cdata, ::SCPCore.CategoricalCovariateDesc, basename)
	categories = categories_spec(cdata)
	combine_vectors_spec(basename, categories; delim='_')
end
extract_covariate_names_spec(data, desc, basename) =
	create_spec(Preprocess(extract_covariate_names), data, fetched(desc), fetched(basename))



function has_centering(::Preprocessing, cov_descs)
	cov_descs = ReproducibleJobs.unsafe_unmanage(cov_descs) # Can we avoid this?
	if cov_descs isa ReadOnly # Can we avoid this?
		cov_descs = cov_descs.value
	end
	any(x->x isa Union{SCPCore.CategoricalCovariateDesc, SCPCore.TwoGroupCovariateDesc}, cov_descs)
end
has_centering_spec(cov_descs) =
	create_spec(Preprocess(has_centering), cov_descs)


function build_designmatrix_dm(::Mat, data, cov_data, cov_descs, ::Any; center, kwargs...)
	cov_data = ReproducibleJobs.unsafe_unmanage(cov_data) # Can we avoid this?
	cov_descs = ReproducibleJobs.unsafe_unmanage(cov_descs) # Can we avoid this?
	if cov_data isa ReadOnly # Can we avoid this?
		cov_data = cov_data.value
	end
	if cov_descs isa ReadOnly # Can we avoid this?
		cov_descs = cov_descs.value
	end

	@assert length(cov_data) == length(cov_descs)
	obs = get_obs_spec(data)

	cm = covariate_matrix_spec.(cov_data, cov_descs; center, kwargs...)
	if center
		ispec = intercept_covariate_matrix_spec(table_nrow_spec(obs))
		hcat_spec(ispec, cm...)
	else
		hcat_spec(cm...)
	end
end
function build_designmatrix_dm(::Obs, ::Any, ::Any, ::Any, cov_names; center, kwargs...)
	cov_names = ReproducibleJobs.unsafe_unmanage(cov_names) # Can we avoid this?
	if cov_names isa ReadOnly # Can we avoid this?
		cov_names = cov_names.value
	end
	center && (cov_names = vcat("Intercept", cov_names))
	create_table_spec("covariate"=>vcat_spec(cov_names...))
end
build_designmatrix_dm(::Var, data, ::Any, ::Any, ::Any; kwargs...) = get_obs_spec(data) # Yes this is correct. (See note below regarding transposing)


# This preprocessing step is needed so that the covariate representations are preprocessed
function build_designmatrix(::Preprocessing, data, cov_data, cov_descs, cov_names; center, kwargs...)
	create_spec(DataMatrixFunction(build_designmatrix_dm), data, cov_data, cov_descs, cov_names; center, kwargs...)
end
build_designmatrix_spec(data, cov_data, cov_descs, cov_names; kwargs...) =
	create_spec(Preprocess(build_designmatrix), data, cov_data, cov_descs, cov_names; kwargs...)



function designmatrix(::Preprocessing, data, args...; center, kwargs...)
	obs = get_obs_spec(data)
	cov_annots, cov_descs = setup_covariate_descriptions(obs, args...)
	cov_data = _extract_data_spec.(Ref(obs), cov_annots)
	center = center || fetched(has_centering_spec(cov_descs))

	cov_basenames = fetched.(_extract_name.(cov_annots))
	cov_names = fetched.(extract_covariate_names_spec.(cov_data, cov_descs, cov_basenames)) # fetch to avoid projecting these
	build_designmatrix_spec(data, cov_data, cov_descs, cov_names; center, kwargs...)
end

# Creates a DataMatrix with obs as var and covariates as obs
# TODO: Consider transposing
# data::DataMatrix, args are covariates (names, two-column DataFrames with IDs+Values, optionally in pairs with covariate descriptions), center::Bool
designmatrix_spec(data, args...; center=true, kwargs...) =
	create_spec(Preprocess(designmatrix), data, args...; center, kwargs...)
function Jobs.designmatrix(data, args...; kwargs...)
	Job(designmatrix_spec(data, args...; kwargs...))
end

