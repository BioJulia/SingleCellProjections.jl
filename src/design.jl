# Yet another new attempt at a better interface
# Making it easier to handle with Specs

detect_covariate_desc(::AbstractVector{<:Union{Missing,Number}}) = SCPCore.numerical_covariate()
detect_covariate_desc(::AbstractVector) = SCPCore.categorical_covariate()

detect_covariate_desc_job(values) = create_job(detect_covariate_desc, values; __version=v"0.1.0")


# TODO: Move this to internal? It can be used in many places.
_extract_data_job(table, column::String) = column_data_job(table, column) # Column in the table (typically obs)
function _extract_data_job(table, annot) # External annotation (DataFrame or spec)
	ids_a = id_column_job(table)
	ids_b = id_column_job(annot)
	ind_job = indexin_job(ids_a, ids_b; not_found=:nothing)
	v = value_column_data_job(annot)
	getindex_or_missing_job(v, ind_job) # The values of the annotation, reordered to match the order in table.
end

_extract_name(column::String) = column
_extract_name(annot) = get_value_colname_job(annot)



function _group_args(desc::SCPCore.TwoGroupCovariateDesc)
	desc.group_b === nothing ? (desc.group_a,) : (desc.group_a,desc.group_b)
end



function setup_covariate_description(obs, a::Pair)
	@assert a.second isa SCPCore.AbstractCovariateDesc
	a
end
function setup_covariate_description(obs, a)
	a => fetched(detect_covariate_desc_job(_extract_data_job(obs, a)))
end

function setup_covariate_descriptions(obs, args...)
	annots = []
	descs = []
	for a in args
		a2 = setup_covariate_description(obs, a)
		push!(annots, a2.first)
		push!(descs, a2.second)
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
twogroup_values_job(v, group_a, args...) =
	create_job(twogroup_values, v, group_a, args...; __version=v"0.1.0")




# intercept_covariate_matrix(n) = trues(n, 1)
intercept_covariate_matrix(n) = ones(n, 1) # Use Float64s because that is more likely to match the eltype of other matrices in the expression



mean_and_scale_job(v; center) = create_job(SCPCore.mean_and_scale, v; center, __version=v"0.1.0")
mean_and_scale_job(v, ::SCPCore.NumericalCovariateDesc; center) = mean_and_scale_job(v; center)
mean_and_scale_job(v, desc::SCPCore.TwoGroupCovariateDesc; center) =
	mean_and_scale_job(twogroup_values_job(v, _group_args(desc)...); center)




# categories_job(v) = unique_job(v) # Doesn't work - because it accepts `missing` as a value
categories_impl(v) = unique(skipmissing(v)) # removes missing and narrows type
categories_job(v) = create_job(categories_impl, v; __version=v"0.1.1")

# TODO: These might be useful if we want support for TwoGroup in pseudobulk
# categories_job(v, desc::SCPCore.CategoricalCovariateDesc) = categories_job(v)
# function categories_job(::Any, desc::SCPCore.TwoGroupCovariateDesc)
# 	if desc.group_b === nothing
# 		[string(desc.group_a), string("!",desc.group_a)] # TODO: Is there a nicer way to handle this? What do we name the "Other" group.
# 	else
# 		[desc.group_a, desc.group_b]
# 	end
# end





function numerical_covariate_matrix_impl(v, (m,s))
	N = length(v)
	x = reshape(v, N, 1) # So we return a N×1 matrix
	(x .- m)./s
end
numerical_covariate_matrix_impl_job(v, ms) =
	create_job(numerical_covariate_matrix_impl, v, ms; __version=v"0.1.0")
function numerical_covariate_matrix(action::Action, data; center)
	ms = fetched(mean_and_scale_job(data; center)) # model - not affected by action
	numerical_covariate_matrix_impl_job(action(data), ms)
end


function categorical_covariate_matrix_impl(ind, n_categories)
	N = length(ind)
	X = falses(N, n_categories)
	X[CartesianIndex.(1:N, ind)] .= true
	X
end
categorical_covariate_matrix_impl_job(ind, n_categories) =
	create_job(categorical_covariate_matrix_impl, ind, n_categories; __version=v"0.1.0")

_too_many_categories_error(len, max_categories) =
	throw(ArgumentError("$len categories in categorical variable, was this intended? Change max_categories (", max_categories, ") if you want to increase the number of allowed categories."))
_too_many_categories_error_job(len, max_categories) = create_job(_too_many_categories_error, len, max_categories)

function categorical_covariate_matrix(action::Action, data; max_categories=100)
	categories = categories_job(data) # model - not affected by action

	# application of model
	data = action(data)
	ind = indexin_job(data, categories)
	result = categorical_covariate_matrix_impl_job(ind, fetched(length_job(categories)))

	len = length_job(categories)
	cond = apply_job(<=(max_categories), len)
	ifelse_pr_job(cond, result, _too_many_categories_error_job(len, max_categories))
end



intercept_covariate_matrix_job(n) = create_job(intercept_covariate_matrix, n; __version=v"0.1.2")
numerical_covariate_matrix_job(data; center) = create_job(Projectable(numerical_covariate_matrix), data; center)
categorical_covariate_matrix_job(data; kwargs...) = create_job(Projectable(categorical_covariate_matrix), data; kwargs...)

twogroup_covariate_matrix(::Preprocessing, data, args...; center) =
	numerical_covariate_matrix_job(twogroup_values_job(data, args...); center)
twogroup_covariate_matrix_job(data, groups...; center) =
	create_job(Preprocess(twogroup_covariate_matrix), data, groups...; center)


covariate_matrix_job(data, desc::SCPCore.NumericalCovariateDesc; center, kwargs...) =
	numerical_covariate_matrix_job(data; center)
function covariate_matrix_job(data, desc::SCPCore.CategoricalCovariateDesc; max_categories=nothing, kwargs...)
	kw = max_categories !== nothing ? (; max_categories) : (;)
	categorical_covariate_matrix_job(data; kw...)
end
covariate_matrix_job(data, desc::SCPCore.TwoGroupCovariateDesc; kwargs...) =
	twogroup_covariate_matrix_job(data, _group_args(desc)...; kwargs...)


extract_covariate_names(::Preprocessing, ::Any, ::SCPCore.NumericalCovariateDesc, basename) = basename
function extract_covariate_names(::Preprocessing, cdata, ::SCPCore.CategoricalCovariateDesc, basename)
	categories = categories_job(cdata)
	combine_vectors_job(basename, categories; delim='_')
end
extract_covariate_names_job(data, desc, basename) =
	create_job(Preprocess(extract_covariate_names), data, fetched(desc), fetched(basename))



function has_centering(::Preprocessing, cov_descs)
	any(x->x isa Union{SCPCore.CategoricalCovariateDesc, SCPCore.TwoGroupCovariateDesc}, cov_descs)
end
has_centering_job(cov_descs) =
	create_job(Preprocess(has_centering), cov_descs)


function build_designmatrix_dm(::Mat, data, cov_data, cov_descs, ::Any; center, kwargs...)
	@assert length(cov_data) == length(cov_descs)
	obs = get_obs_job(data)

	cm = covariate_matrix_job.(cov_data, cov_descs; center, kwargs...)
	if center
		ispec = intercept_covariate_matrix_job(table_nrow_job(obs))
		hcat_job(vcat(ispec, cm))
	else
		hcat_job(cm)
	end
end
function build_designmatrix_dm(::Obs, ::Any, ::Any, ::Any, cov_names; center, kwargs...)
	center && (cov_names = vcat("Intercept", cov_names))
	# create_table_job("covariate"=>vcat_job(cov_names...))
	create_table_job("covariate"=>vcat_job(cov_names))
end
build_designmatrix_dm(::Var, data, ::Any, ::Any, ::Any; kwargs...) = get_obs_job(data) # Yes this is correct. (See note below regarding transposing)


# This preprocessing step is needed so that the covariate representations are preprocessed
function build_designmatrix(::Preprocessing, data, cov_data, cov_descs, cov_names; center, kwargs...)
	create_job(DataMatrixFunction(build_designmatrix_dm), data, cov_data, cov_descs, cov_names; center, kwargs...)
end
build_designmatrix_job(data, cov_data, cov_descs, cov_names; kwargs...) =
	create_job(Preprocess(build_designmatrix), data, cov_data, cov_descs, cov_names; kwargs...)



function designmatrix(::Preprocessing, data, args...; center, kwargs...)
	obs = get_obs_job(data)
	cov_annots, cov_descs = setup_covariate_descriptions(obs, args...)
	cov_data = _extract_data_job.(Ref(obs), cov_annots)
	center = center || fetched(has_centering_job(cov_descs))

	cov_basenames = fetched.(_extract_name.(cov_annots))
	cov_names = fetched.(extract_covariate_names_job.(cov_data, cov_descs, cov_basenames)) # fetch to avoid projecting these
	build_designmatrix_job(data, cov_data, cov_descs, cov_names; center, kwargs...)
end

# Creates a DataMatrix with obs as var and covariates as obs
# TODO: Consider transposing
# data::DataMatrix, args are covariates (names, two-column DataFrames with IDs+Values, optionally in pairs with covariate descriptions), center::Bool
designmatrix_job(data, args...; center=true, kwargs...) =
	create_job(Preprocess(designmatrix), data, args...; center, kwargs...)
function Jobs.designmatrix(data, args...; kwargs...)
	designmatrix_job(data, args...; kwargs...)
end

