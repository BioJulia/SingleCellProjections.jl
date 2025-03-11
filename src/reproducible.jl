ReproducibleJobs.unmanage_rec(x::SCPC.ValueVector) = x
ReproducibleJobs.unmanage_rec(x::SCPC.ProjectionModel) = x







# --- Internal checkpoints (asserts at the spec level) -------------------------
# function is_id_subset(a,b)
# 	id_col = only(names(a,1))
# 	@assert only(names(b,1)) == id_col
# 	@assert size(a,2) == 1
# 	@assert size(b,2) == 1

# 	issubset(a[!,1],b[!,1])
# end
# is_id_subset_spec(a, b) = create_spec(is_id_subset, a, b; use_cache=false, __version=v"0.1.0")

# function check_missing_ids_spec(value, ids, ids2)
# 	cond_spec = is_id_subset_spec(ids2, ids)
# 	error_spec = "Found missing IDs" # TODO: use spec so we can show names of missing IDs
# 	ifelse_spec(cond_spec, value, error_spec)
# 	# value
# end



# --- Internal specs, used when forwarding other specs -------------------------


function intersect_ids_impl(ids::DataFrame, ids2::DataFrame)
	id_col = only(names(ids,1))
	@assert only(names(ids2,1)) == id_col
	@assert size(ids,2) == 1
	@assert size(ids2,2) == 1
	DataFrame(id_col => intersect(ids[!,1],ids2[!,1]))
end

# function intersect_ids(action::Action, ids, ids2)
# 	ids = action(ids)
# 	ids2 = action(ids2)
# 	ids === ids2 && return ids

# 	spec = create_spec(intersect_ids_impl, ids, ids2; use_cache=false, __version=v"0.1.0")

# 	# Here's the place for checking for missing and extra IDs!
# 	check_missing_ids_spec(spec, ids, ids2)
# end
# create_intersect_ids_spec(ids, ids2) =
# 	create_spec(Projectable(intersect_ids), ids, ids2)



function find_matching_ids(action::Action, f, df; project_ids::Symbol)
	@assert project_ids in (:no,:yes,:intersect)
	if project_ids == :no
		f = action(f)
		df = action(df)
	end
	spec = create_spec(SCPC.find_matching_ids, f, df; use_cache=true, __version=v"0.1.0")

	if project_ids == :intersect
		df = action(df)
		# TODO: simplify ids2 spec by using a function for extracting IDs directly
		ids2 = create_spec(SCPC.find_matching_ids, Returns(true), df; use_cache=false, __version=v"0.1.0")
		spec = create_spec(intersect_ids_impl, spec, ids2; use_cache=true, __version=v"0.1.0")
	end
	spec
end


create_find_matching_ids_spec(f, df; project_ids) =
	create_spec(Projectable(find_matching_ids), f, df; project_ids)
Jobs.find_matching_ids(args...; kwargs...) =
	Job(create_find_matching_ids_spec(args...; kwargs...))




ids_to_indices(action::Action, args...) =
	create_spec(SCPC.ids_to_indices, action(args)...; use_cache=true, __version=v"0.1.0")
create_ids_to_indices_spec(df, ids) =
	create_spec(Projectable(ids_to_indices), df, ids)

annotation_getindex(action::Action, args...) =
	create_spec(SCPC.annotation_getindex, action(args)...; use_cache=true, __version=v"0.1.0")
create_annotation_getindex_spec(df, ind) =
	create_spec(Projectable(annotation_getindex), df, ind; use_cache=false)



matrix_getindex(action::Action, args...; kwargs...) =
	create_spec(SCPC.matrix_getindex, action(args)...; action(kwargs)..., use_cache=false, __version=v"0.1.0")
function create_matrix_getindex_spec(data; kwargs...)
	isempty(setdiff(keys(kwargs), (:var_ind,:obs_ind))) || throw(ArgumentError("Only allowed kwargs are `var_ind` and `obs_ind`, got: $(keys(kwargs))."))
	create_spec(Projectable(matrix_getindex), data; kwargs...)
end


extract_annotation(action::Action, args...) =
	create_spec(SCPC.extract_annotation, action(args)...; use_cache=false, __version=v"0.1.0")
create_extract_annotation_spec(df, name) =
	create_spec(Projectable(extract_annotation), df, name)


annotation_nrows_impl(df) = size(df,1)
annotation_nrows(action::Action, df) =
	create_spec(annotation_nrows_impl, action(df); use_cache=false, __version=v"0.1.0")
annotation_nrows_spec(df) =
	create_spec(Projectable(annotation_nrows), df)


# TODO: Rename
hcat_impl(action::Action, args...) =
	create_spec(LinearAlgebra.hcat, action(args)...; use_cache=false, __version=v"0.1.0")
create_hcat_spec(args...) = create_spec(Projectable(hcat_impl), args...)



# NB: This assumes that the caller knows that `vals` exactly matches the ID column in `df`.
add_column_impl(df::DataFrame, name, vals) = insertcols(df, name=>vals; copycols=false)

add_column(action::Action, df, name, vals) =
	create_spec(add_column_impl, action(df), name, action(vals); use_cache=false, __version=v"0.1.0")
create_add_column_spec(df, name, vals) = create_spec(Projectable(add_column), df, name, vals)

# ------------------------------------------------------------------------------



# --- DataMatrix triplets ------------------------------------------------------

# low-level specs for loading counts
function load_var_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_features(filename, DataFrame)
end
load_var_spec(filename) = create_spec(load_var_impl, filename; use_cache=false, __version=v"0.1.0")

function load_obs_impl(filename)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SingleCell10x.read10x_barcodes(filename, DataFrame)
end
load_obs_spec(filename; kwargs...) = create_spec(load_obs_impl, filename; use_cache=false, kwargs..., __version=v"0.1.0")

combine_obs_spec(obs, sample_names; id_col="cell_id", id_delim='_', kwargs...) =
	create_spec(SCPC.combine_obs, obs, sample_names; id_col, id_delim, use_cache=true, kwargs..., __version=v"0.1.0") # TODO: Do we want use_cache=true here? Probably.

combine_var_spec(vars; kwargs...) = create_spec(SCPC.combine_var, vars; use_cache=true, kwargs..., __version=v"0.1.0") # TODO: Do we want use_cache=true here? Probably.

sample_var_indices_spec(sample_var, var; kwargs...) =
	create_spec(SCPC.sample_var_indices, sample_var, var; use_cache=true, kwargs..., __version=v"0.1.0")

function load_sample_matrix_metadata_impl(filename, var_ind)
	@assert filename isa ChecksummedFilePath
	filename = string(filename)
	SCPC.load_sample_matrix_metadata(filename, var_ind)
end
load_sample_matrix_metadata_spec(filename, var_ind; kwargs...) =
	create_spec(load_sample_matrix_metadata_impl, filename, var_ind; use_cache=true, kwargs..., __version=v"0.1.0")

function load_hcat_sample_matrices_impl(filenames, args...; kwargs...)
	@assert all(x->x isa ChecksummedFilePath, filenames)
	filenames = string.(filenames)
	SCPC.load_hcat_sample_matrices(filenames, args...; kwargs...)
end
load_hcat_sample_matrices_spec(filenames, matrix_metadatas, var_inds; kwargs...) =
	create_spec(load_hcat_sample_matrices_impl, filenames, matrix_metadatas, var_inds; use_cache=false, kwargs..., __version=v"0.1.0")




function load_counts(f::Union{Mat,Var}, filename_specs; sample_names, prefilter, extra_id_cols)
	sample_var_specs = load_var_spec.(filename_specs)
	var_spec = combine_var_spec(sample_var_specs; prefilter, extra_id_cols)

	if f isa Var
		return var_spec
	else # if f isa Mat
		var_ind_specs = prefetch.(sample_var_indices_spec.(sample_var_specs, var_spec; extra_id_cols))
		metadata_specs = prefetch.(load_sample_matrix_metadata_spec.(filename_specs, var_ind_specs))
		return load_hcat_sample_matrices_spec(filename_specs, metadata_specs, var_ind_specs)
	end
end
function load_counts(::Obs, filename_specs; sample_names, prefilter, extra_id_cols)
	sample_obs_specs = load_obs_spec.(filename_specs)
	combine_obs_spec(sample_obs_specs, sample_names)
end


function Jobs.load_counts(filenames;
                          sample_names,
                          prefilter = "feature_type"=>isequal("Gene Expression"),
                          extra_id_cols = "feature_type",
                          kwargs...)
	filenames isa AbstractArray || (filenames = [filenames])
	sample_names isa AbstractArray || (sample_names = [sample_names])

	# TODO: Support .mtx.gz, with the added complication that we need to find matching
	#       feature/barcode files already here so that they can be checksummed too.
	@assert all(x->lowercase(splitext(x)[2])==".h5", filenames) "Only 10x .h5 files are currently supported"

	filename_jobs = checksummedfilepath_job.(filenames)
	Job(create_spec(DataMatrixFunc(load_counts), filename_jobs; sample_names, prefilter, extra_id_cols, use_cache=false, kwargs...))
end



# TODO: Find a better name?
function annot_leftjoin_impl(annot::DataFrame, df)
	id_col = only(names(annot,1))
	df_id_col = only(names(df,1))
	@assert id_col == df_id_col "Annotation IDs didn't match, got \"$df_id_col\", but expected \"$id_col\"."
	annot = copy(annot; copycols=false)
	leftjoin!(annot, df; on=id_col)
end
annot_leftjoin(action::Action, args...) =
	create_spec(annot_leftjoin_impl, action(args)...; use_cache=false, __version=v"0.1.0")



annotate(::Mat, data; kwargs...) = get_matrix_spec(data)
function annotate(f::Union{Var,Obs}, data; kwargs...)
	s = get_spec(f, data)
	df = get(kwargs, f isa Var ? :var : :obs, nothing)
	df === nothing && return s
	return create_spec(Projectable(annot_leftjoin), s, df)
end

# These should perhaps have a parameter saying how projections should be handled.
# Because modifications to base var we probably also want in proj var.
# Or maybe this will be solved differently, some projection step might choose to replace proj var with base var, and then it's not a problem.
Jobs.annotate_obs(data, df; kwargs...) =
	Job(create_spec(DataMatrixFunc(annotate), data; use_cache=false, kwargs..., obs=df))
Jobs.annotate_var(data, df; kwargs...) =
	Job(create_spec(DataMatrixFunc(annotate), data; use_cache=false, kwargs..., var=df))
Jobs.annotate(data, var_df, obs_df; kwargs...) =
	Job(create_spec(DataMatrixFunc(annotate), data; use_cache=false, kwargs..., var=var_df, obs=obs_df))




function _subset_ind(f::Union{Var,Obs}, data; kwargs...)
	ids = get(kwargs, f isa Var ? :var_ids : :obs_ids, nothing)
	ids === nothing && return nothing
	s = get_spec(f, data)
	create_ids_to_indices_spec(s, ids)
end

function subset_matrix(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	var_ind = _subset_ind(Var(), data; kwargs...)
	obs_ind = _subset_ind(Obs(), data; kwargs...)
	if var_ind !== nothing && obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind), obs_ind=prefetch(obs_ind))
	elseif var_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind))
	elseif obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; obs_ind=prefetch(obs_ind))
	else
		matrix_spec
	end
end
function subset_matrix(f::Union{Var,Obs}, data; kwargs...)
	s = get_spec(f, data)
	ind = _subset_ind(f, data; kwargs...)
	ind === nothing && return s
	create_annotation_getindex_spec(s, prefetch(ind))
end


Jobs.subset_matrix(data, var_ids, obs_ids; kwargs...) =
	Job(create_spec(DataMatrixFunc(subset_matrix), data; use_cache=false, kwargs..., var_ids, obs_ids))
Jobs.subset_var(data, var_ids; kwargs...) =
	Job(create_spec(DataMatrixFunc(subset_matrix), data; use_cache=false, kwargs..., var_ids))
Jobs.subset_obs(data, obs_ids; kwargs...) =
	Job(create_spec(DataMatrixFunc(subset_matrix), data; use_cache=false, kwargs..., obs_ids))





function _filter_ind(f::Union{Var,Obs}, data; kwargs...)
	fun = get(kwargs, f isa Var ? :fvar : :fobs, nothing)
	fun === nothing && return nothing
	s = get_spec(f, data)
	project_ids = kwargs[f isa Var ? :project_var_ids : :project_obs_ids]
	id_spec = create_find_matching_ids_spec(fun, s; project_ids)
	create_ids_to_indices_spec(s, id_spec)
end

function filter_matrix(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	var_ind = _filter_ind(Var(), data; kwargs...)
	obs_ind = _filter_ind(Obs(), data; kwargs...)
	if var_ind !== nothing && obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind), obs_ind=prefetch(obs_ind))
	elseif var_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; var_ind=prefetch(var_ind))
	elseif obs_ind !== nothing
		create_matrix_getindex_spec(matrix_spec; obs_ind=prefetch(obs_ind))
	else
		matrix_spec
	end
end
function filter_matrix(f::Union{Var,Obs}, data; kwargs...)
	s = get_spec(f, data)
	ind = _filter_ind(f, data; kwargs...)
	ind === nothing && return s
	create_annotation_getindex_spec(s, prefetch(ind))
end



# Can we share more code with subset_ functions above?
Jobs.filter_matrix(fvar, fobs, data; project_var_ids=:intersect, project_obs_ids=:no, kwargs...) =
	Job(create_spec(DataMatrixFunc(filter_matrix), data; use_cache=false, kwargs..., fvar, fobs, project_var_ids, project_obs_ids))
Jobs.filter_var(fvar, data; project_ids=:intersect, kwargs...) =
	Job(create_spec(DataMatrixFunc(filter_matrix), data; use_cache=false, kwargs..., fvar, project_var_ids=project_ids))
Jobs.filter_obs(fobs, data; project_ids=:no, kwargs...) =
	Job(create_spec(DataMatrixFunc(filter_matrix), data; use_cache=false, kwargs..., fobs, project_obs_ids=project_ids))


# ------------------------------------------------------------------------------


var_counts_fraction_impl(action::Action, counts, sub_ind, tot_ind) =
	create_spec(SCPC.var_counts_fraction2, action(counts), action(sub_ind), action(tot_ind); use_cache=true, __version=v"0.1.0")
create_var_counts_fraction_impl_spec(counts, sub_ind, tot_ind) = create_spec(Projectable(var_counts_fraction_impl), counts, sub_ind, tot_ind)

var_counts_fraction(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
var_counts_fraction(::Var, counts, args...; kwargs...) = get_var_spec(counts)
function var_counts_fraction(::Obs, counts, sub_filter, tot_filter, col; project_ids)
	sub_ind = prefetch(_filter_ind(Var(), counts; fvar=sub_filter, project_var_ids=project_ids))
	tot_ind = prefetch(_filter_ind(Var(), counts; fvar=tot_filter, project_var_ids=project_ids))
	values_spec = create_var_counts_fraction_impl_spec(get_matrix_spec(counts), sub_ind, tot_ind)
	create_add_column_spec(get_obs_spec(counts), col, values_spec)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_fraction(counts, sub_filter, tot_filter, col; project_ids=:intersect)
	Job(create_spec(DataMatrixFunc(var_counts_fraction), counts, sub_filter, tot_filter, col; use_cache=false, project_ids))
end



var_counts_sum_impl(action::Action, f, counts, ind) =
	create_spec(SCPC.var_counts_sum2, f, action(counts), action(ind); use_cache=true, __version=v"0.1.0")
create_var_counts_sum_impl_spec(f, counts, ind) = create_spec(Projectable(var_counts_sum_impl), f, counts, ind)

var_counts_sum(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
var_counts_sum(::Var, counts, args...; kwargs...) = get_var_spec(counts)
function var_counts_sum(::Obs, counts, filter, col; project_ids, f=identity)
	ind = prefetch(_filter_ind(Var(), counts; fvar=filter, project_var_ids=project_ids))
	values_spec = create_var_counts_sum_impl_spec(f, get_matrix_spec(counts), ind)
	create_add_column_spec(get_obs_spec(counts), col, values_spec)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_sum(f, counts, filter, col; project_ids=:intersect)
	Job(create_spec(DataMatrixFunc(var_counts_sum), counts, filter, col; use_cache=false, f, project_ids))
end
Jobs.var_counts_sum(counts, filter, col; kwargs...) = Jobs.var_counts_sum(identity, counts, filter, col; kwargs...)





function logtransform_matrix(action::Action, T::DataType, matrix; scale_factor, var_ind)
	matrix = action(matrix)
	var_ind = action(var_ind)
	create_spec(SCPC.logtransform_matrix, T, matrix; scale_factor, var_ind, use_cache=false, __version=v"0.1.0")
end

function logtransform(f::Union{Mat,Var}, T::DataType, data; var_filter=Returns(true), project_var_ids=:intersect, kwargs...)
	var_spec = get_var_spec(data)

	# # --- Old ---
	# # var_ids = create_find_matching_ids_spec(var_filter, var_spec; fix_ids=true)
	# var_ids = create_find_matching_ids_spec(var_filter, var_spec; project_ids=:intersect) # project_ids should be param to logtransform

	# # all_var_ids = create_find_matching_ids_spec(Returns(true), var_spec; fix_ids=false)
	# all_var_ids = create_find_matching_ids_spec(Returns(true), var_spec; project_ids=:no)

	# # All IDs in matching base case, in the same order as in base
	# # intersected_var_ids_spec = create_intersect_ids_spec(var_ids)
	# intersected_var_ids_spec = create_intersect_ids_spec(var_ids, all_var_ids)
	# # ---------

	var_ids = create_find_matching_ids_spec(var_filter, var_spec; project_ids=project_var_ids)
	var_ind = prefetch(create_ids_to_indices_spec(var_spec, var_ids))

	if f isa Var
		create_annotation_getindex_spec(var_spec, var_ind)
	else # if f isa Mat
		matrix_spec = get_matrix_spec(data)
		create_spec(Projectable(logtransform_matrix), T, matrix_spec; use_cache=false, var_ind, kwargs...)
	end
end
logtransform(::Obs, ::DataType, data; kwargs...) = get_obs_spec(data)


function Jobs.logtransform(T::DataType, counts; scale_factor=10_000, kwargs...)
	Job(create_spec(DataMatrixFunc(logtransform), T, counts; use_cache=false, scale_factor, kwargs...))
end
Jobs.logtransform(counts; kwargs...) = Jobs.logtransform(Float64, counts; kwargs...)



# ------------------------------------------------------------------------------

# It's confusing with center_matrix, Jobs.center_matrix and SCPC.center_matrix being different functions having the same name.
# But we'll get rid of it so that doesn't matter!
function center_matrix(action::Action, matrix)
	# model is created from original data
	model = create_spec(SCPC.CenteringModel2, matrix; use_cache=true, __version=v"0.1.0")
	create_spec(SCPC.center_project, model, action(matrix); __version=v"0.1.0")
end


function center_matrix(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(center_matrix), matrix_spec; use_cache=false, kwargs...)
end
center_matrix(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)


# TEMP, use this as a simple example for testing out projections and specs
function Jobs.center_matrix(args...; kwargs...)
	Job(create_spec(DataMatrixFunc(center_matrix), args...; use_cache=false, kwargs...))
end


# ------------------------------------------------------------------------------

value_vector_model_spec(annot; kwargs...) =
	create_spec(SCPC.value_vector_model, annot; use_cache=true, kwargs..., __version=v"0.1.0")


function value_vector(action::Action, annot; kwargs...)
	model = value_vector_model_spec(annot; kwargs...)
	create_spec(SCPC.value_vector_project, model, action(annot); __version=v"0.1.0")
end
value_vector_spec(annot; kwargs...) =
	create_spec(Projectable(value_vector), annot; use_cache=true, kwargs...)


function covariate(action::Action, args...; kwargs...)
	model = create_spec(SCPC.covariate_model, args...; use_cache=true, kwargs..., __version=v"0.1.0")
	create_spec(SCPC.covariate_project, model, action(args)...; __version=v"0.1.0")
end
covariate_spec(args...; kwargs...) =
	create_spec(Projectable(covariate), args...; kwargs...)



# TODO: Move these into SingleCellProjections.jl?
function _add_covariate_names!(out, name, model::SCPC.CategoricalValueVectorModel)
	for c in model.categories
		push!(out, string(name, '_', c))
	end
end
_add_covariate_names!(out, name, ::SCPC.NumericalValueVectorModel) = push!(out, name)
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
	create_spec(covariate_names_impl, action(args)...; center, use_cache=false, __version=v"0.1.0")
covariate_names_spec(args...; center) =
	create_spec(Projectable(covariate_names), args...; center)



function design(f::Union{Mat,Obs}, data, args...; center)
	# TODO: early out if only centering or doing nothing at all

	obs = get_obs_spec(data)
	annotation_specs = [create_extract_annotation_spec(obs, a) for a in args]

	if f isa Mat
		value_vector_specs = prefetch.(value_vector_spec.(annotation_specs))
		covariate_specs = covariate_spec.(value_vector_specs; center)

		if center
			nrows_spec = prefetch(annotation_nrows_spec(obs))
			intercept_spec = covariate_spec(nrows_spec; center)
			pushfirst!(covariate_specs, intercept_spec)
		end
		return create_hcat_spec(covariate_specs...)
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
	create_spec(DataMatrixFunc(design), data, args...; use_cache=false, center, kwargs...)
function Jobs.designmatrix(data, args...; kwargs...)
	Job(designmatrix_spec(data, args...; kwargs...))
end







negative_regression_matrix_impl(::Action, data, dm; kwargs...) =
	create_spec(SCPC.negative_regression_matrix, data, dm; use_cache=true, kwargs..., __version=v"0.1.0") # NB: No action, always use original
negative_regression_matrix_impl_spec(data, dm; kwargs...) =
	create_spec(Projectable(negative_regression_matrix_impl), data, dm; kwargs...)


function negative_regression_matrix(::Mat, data, dm; kwargs...)
	negative_regression_matrix_impl_spec(get_matrix_spec(data), get_matrix_spec(dm); kwargs...)
end
negative_regression_matrix(::Var, data, dm; kwargs...) = get_spec(Var(), data)
negative_regression_matrix(::Obs, data, dm; kwargs...) = get_spec(Obs(), dm)


function negative_regression_matrix_spec(data, dm; rtol=nothing)
	rtol = @something rtol sqrt(eps())
	create_spec(DataMatrixFunc(negative_regression_matrix), data, dm; use_cache=true, rtol)
end
function Jobs.negative_regression_matrix(args...; kwargs...)
	Job(negative_regression_matrix_spec(args...; kwargs...))
end






normalize_matrix_impl(action::Action, data, negβT, dm) =
	create_spec(SCPC.normalize_matrix2, action(data), action(negβT), action(dm); use_cache=false, __version=v"0.1.0")
normalize_matrix_impl_spec(data, negβT, dm) =
	create_spec(Projectable(normalize_matrix_impl), data, negβT, dm; use_cache=false)

function normalize_matrix(::Mat, data, args...; center=true, rtol=nothing)
	# Maybe we should go for the matrix specs directly?
	dm = designmatrix_spec(data, args...; center)
	negβT = negative_regression_matrix_spec(data, dm; rtol)
	normalize_matrix_impl_spec(get_matrix_spec(data), get_matrix_spec(negβT), get_matrix_spec(dm))
end
normalize_matrix(f::Union{Var,Obs}, data, args...; center=true) = get_spec(f, data)


function Jobs.normalize_matrix(data, args...; center=true, kwargs...)
	Job(create_spec(DataMatrixFunc(normalize_matrix), data, args...; use_cache=false, center, kwargs...))
end


# ------------------------------------------------------------------------------


# SVD is an example where the model comes after the result. I.e. svd(data) => UΣVᵀ, but the model is just UΣ.
function svd(action::Action, matrix; kwargs...)
	# First SVD of unprojected
	svd_spec = create_spec(SCPC.implicitsvd, matrix; use_cache=true, kwargs..., __version=v"0.1.0")

	if action isa Eval
		return svd_spec
	else# if action isa Projection
		return create_spec(SCPC.svd_project, svd_spec, action(matrix); use_cache=true, __version=v"0.1.0")
	end
end


function svd(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(svd), matrix_spec; kwargs...)
end
svd(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)

function Jobs.svd(args...; seed=1234, kwargs...)
	Job(create_spec(DataMatrixFunc(svd), args...; seed, kwargs...))
end


# ------------------------------------------------------------------------------


function adjoint_matrix(action::Action, X)
	# TODO: Should we do unwrapping of adjoint(adjoint(X)) here as well?
	create_spec(LinearAlgebra.adjoint, action(X); use_cache=false, __version=v"0.1.0")
end

create_adjoint_matrix_spec(X) =
	create_spec(Projectable(adjoint_matrix), X)

function adjoint_impl(::Mat, data)
	if data.f == DataMatrixFunc(adjoint_impl)
		get_matrix_spec(data.args[1]) # adjoint(adjoint(X)) == X
	else
		create_adjoint_matrix_spec(get_matrix_spec(data))
	end
end
adjoint_impl(::Var, data) = get_obs_spec(data)
adjoint_impl(::Obs, data) = get_var_spec(data)

# NB: We call it transpose even though we use adjoint internally.
#     Because a user is more likely to use data' than transpose(data) even when they mean transposing.
function Jobs.transpose(data)
	Job(create_spec(DataMatrixFunc(adjoint_impl), data; use_cache=false))
end
