_splattable(x::Union{Tuple,AbstractVector}) = x
_splattable(x) = (x,)



"""
	nonmissing_ind(column_data...)

Computes indices of non-missing rows in `column_data`.
Each entry in `column_data` must be a vector and the lengths must match.

If there are no columns, or there are no missing values, `:` is returned.
Otherwise a sorted `Vector{Int}` with indices of the non-missing rows.
"""
function nonmissing_ind(column_data...)
	isempty(column_data) && return Colon()

	n = length(first(column_data))
	all(c->length(c)==n, column_data) || throw(ArgumentError("All extracted columns must have the same length."))

	mask = trues(n)
	for c in column_data
		mask .&= .!ismissing.(c)
	end

	all(mask) && return Colon()
	findall(mask)
end
nonmissing_ind_spec(column_data...) =
	create_spec(nonmissing_ind, column_data...; __version=v"0.1.0")


function ftest_table_pr(action::Action, matrix, var_ids, h1_design, h0_design)
	cached(create_spec(SCPCore.ftest_table2,
	                   action(matrix), action(var_ids), action(h1_design), action(h0_design);
	                   __version=v"0.0.1"))
end

ftest_table_spec(matrix, var_ids, h1_design, h0_design) =
	create_spec(Projectable(ftest_table_pr), matrix, var_ids, h1_design, h0_design)


function ftest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing)
	# TODO: Filter observables with missing values (according to policy set in kwargs)

	extra_kwargs = max_categories === nothing ? (;) : (; max_categories)

	# Wrap single hypothesis in tuples so we can splat them below
	h1 = _splattable(h1)
	h0 = _splattable(h0)

	# Hmm. We want h1 to be mean-zero (if center=true), but we don't want the intercept column.
	h1_design = designmatrix_spec(data, h1...; center=false, extra_kwargs...)
	h0_design = designmatrix_spec(data, h0...; center, extra_kwargs...)

	matrix = get_matrix_spec(data)
	var_ids = id_column_spec(get_var_spec(data))

	ftest_table_spec(matrix, var_ids, get_matrix_spec(h1_design), get_matrix_spec(h0_design))
end


ftest_spec(data, h1; kwargs...) =
	create_spec(Preprocess(ftest), data, h1; kwargs...)
function Jobs.ftest(data, h1; kwargs...)
	Job(ftest_spec(data, h1; kwargs...))
end





function ttest_table_pr(action::Action, matrix, var_ids, h1_design, h1_scale, h0_design)
	cached(create_spec(SCPCore.ttest_table2,
	                   action(matrix), action(var_ids),
	                   action(h1_design), prefetched(action(h1_scale)),
	                   action(h0_design);
	                   __version=v"0.0.1"))
end

ttest_table_spec(matrix, var_ids, h1_design, h1_scale, h0_design) =
	create_spec(Projectable(ttest_table_pr), matrix, var_ids, h1_design, h1_scale, h0_design)


# function ttest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing, h1_missing=:skip, h0_missing=:error)
# 	# Something like this
# 	@assert h1_missing in (:skip,:error)
# 	@assert h0_missing in (:skip,:error)

# 	# # Filter observables with missing values if desired
# 	# if h1_missing == :skip || h0_missing == :skip
# 	#  or should this be done in covariate_stages?
# 	#  nah, not possible, because we need for both h0 and h1 potentially
# 	#  so we refactor to first get covariate_descriptions
# 	#  then use that to filter if desired
# 	#  	(NB: ensure `:` results in a no-op filter that just returns the parent datamatrix)
# 	#  and to create design matrix
# 	# end


# 	# Wrap single hypothesis in tuples so we can splat them below
# 	h0 = _splattable(h0)


# 	h0_covariate_descriptions, center = setup_covariate_descriptions(h0...; center)

# 	# TODO: We want h1 to be mean-zero (if center=true), but we don't want the intercept column. Fix.
# 	h1_covariate_descriptions, _ = setup_covariate_descriptions(h1; center=false)

# 	# Handle missing values
# 	skip_missing_cols = []
# 	let obs = get_obs_spec(data)
# 		if h1_missing == :skip
# 			col, desc = only(h1_covariate_descriptions)
# 			push!(skip_missing_cols, _value_vector_data_spec(obs, col, desc))
# 		end
# 		if h0_missing == :skip
# 			for (col,desc) in h0_covariate_descriptions
# 				desc === SCPCore.intercept_covariate() && continue # an intercept does not have missing values
# 				push!(skip_missing_cols, _value_vector_data_spec(obs, col, desc))
# 			end
# 		end
# 	end

# 	if !isempty(skip_missing_cols)
# 		# @show skip_missing_cols
# 		obs_ind = nonmissing_ind_spec(skip_missing_cols...)
# 		data = create_datamatrix_getindex_spec(data; obs_ind)
# 	end




# 	extra_args = max_categories === nothing ? (;) : (; max_categories)


# 	# We need to get the covariate model spec somewhere.
# 	# Because it contains the scale info needed for the ttest (for difference).

# 	obs = get_obs_spec(data)

# 	(; covariate_model_specs, covariate_specs, covariate_names) = covariate_stages(obs, h1_covariate_descriptions; center=false, extra_args...)
# 	h1_covariate_model = only(covariate_model_specs)
# 	h1_scale = covariate_scale_spec(h1_covariate_model)
# 	h1_design = build_designmatrix_spec(data, covariate_specs, covariate_names)

# 	h0_design = designmatrix_spec(data, h0...; center, extra_args...) # TODO: Use h0_covariate_descriptions and build_design_matrix?

# 	matrix = get_matrix_spec(data)
# 	var_ids = id_column_spec(get_var_spec(data))

# 	ttest_table_spec(matrix, var_ids, get_matrix_spec(h1_design), h1_scale, get_matrix_spec(h0_design))
# end

# function ttest_setup(::Preprocessing, data, h1; )


# TODO: This does not work properly with projections. Fix.
function ttest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing, h1_missing=:skip, h0_missing=:error)
	@assert h1_missing in (:skip,:error)
	@assert h0_missing in (:skip,:error)

	# Wrap single hypothesis in tuples so we can splat them below
	h0 = _splattable(h0)

	# Check that h1 is of an allowed kind of test
	if h1 isa Pair
		let (_,desc)=h1
			desc isa Union{SCPCore.NumericalCovariateDesc,SCPCore.TwoGroupCovariateDesc} || error("h1 must be a numerical or twogroup covariate, got $(typeof(desc)).")
		end
	else
		h1 = h1=>numerical_covariate() # default to numerical if not given - we want something 1d
	end


	# Handle missing values
	skip_missing_cols = []
	let obs = get_obs_spec(data)
		if h1_missing == :skip
			push!(skip_missing_cols, _extract_data_spec(obs, h1.first))
		end
		if h0_missing == :skip
			for a in h0
				if a isa Pair
					a = a.first
				end
				push!(skip_missing_cols, _extract_data_spec(obs, a))
			end
		end
	end

	if !isempty(skip_missing_cols)
		# @show skip_missing_cols
		obs_ind = nonmissing_ind_spec(skip_missing_cols...)
		data = create_datamatrix_getindex_spec(data; obs_ind)
	end



	obs = get_obs_spec(data)


	extra_kwargs = max_categories === nothing ? (;) : (; max_categories)


	h1_cov_annot, h1_cov_desc = h1


	center = center || (h1_cov_desc isa TwoGroupCovariateDesc) # Center if h1 requires it
	if !center # Figure out if h0 requires centering
		_, h0_cov_descs = setup_covariate_descriptions_new(obs, h0...)
		center = fetched(has_centering_spec(h0_cov_descs))
	end


	h0_design = designmatrix_spec(data, h0...; center, extra_kwargs...)

	h1_cov_data = _extract_data_spec(obs, h1_cov_annot)
	ms = mean_and_scale_spec(h1_cov_data, h1_cov_desc; center)
	h1_scale = fetched(getindex_spec(ms, 2))
	h1_design_mat = covariate_matrix_new_spec(h1_cov_data, h1_cov_desc; center) # center affects this column, but we don't get an intercept

	matrix = get_matrix_spec(data)
	var_ids = id_column_spec(get_var_spec(data))

	ttest_table_spec(matrix, var_ids, h1_design_mat, h1_scale, get_matrix_spec(h0_design))
end



ttest_spec(data, h1; kwargs...) =
	create_spec(Preprocess(ttest), data, h1; kwargs...)
function Jobs.ttest(data, h1; kwargs...)
	Job(ttest_spec(data, h1; kwargs...))
end
