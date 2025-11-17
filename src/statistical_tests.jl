_splattable(x::Union{Tuple,AbstractVector}) = x
_splattable(x) = (x,)



"""
	non_missing_ind_impl(column_data...)

Computes indices of non-missing rows in `column_data`.
Each entry in `column_data` must be a vector and the lengths must match.

If there are no columns, or there are no missing values, `:` is returned.
Otherwise a sorted `Vector{Int}` with indices of the non-missing rows.
"""
function non_missing_ind_impl(column_data...)
	isempty(column_data) && return Colon()

	n = length(first(column_data))
	all(c->length(c)==n, column_data) || throw(ArgumentError("All extracted columns must have the same length."))

	mask = trues(n)
	for c in column_data
		mask .|= .!ismissing.(c)
	end

	all(mask) && return Colon()
	findall(mask)
end
non_missing_ind(action::Action, column_data...) =
	create_spec(non_missing_ind_impl, action(column_data)...; __version=v"0.0.1")
non_missing_ind_spec(column_data...) =
	create_spec(Projectable(non_missing_ind), column_data...)


function ftest_table_pr(action::Action, matrix, var_ids, h1_design, h0_design)
	cached(create_spec(SCPCore.ftest_table2,
	                   action(matrix), action(var_ids), action(h1_design), action(h0_design);
	                   __version=v"0.0.1"))
end

ftest_table_spec(matrix, var_ids, h1_design, h0_design) =
	create_spec(Projectable(ftest_table_pr), matrix, var_ids, h1_design, h0_design)


function ftest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing)
	# TODO: Filter observables with missing values (according to policy set in kwargs)

	extra_args = max_categories === nothing ? (;) : (; max_categories)

	# Wrap single hypothesis in tuples so we can splat them below
	h1 = _splattable(h1)
	h0 = _splattable(h0)

	# Hmm. We want h1 to be mean-zero (if center=true), but we don't want the intercept column.
	h1_design = designmatrix_spec(data, h1...; center=false, extra_args...)
	h0_design = designmatrix_spec(data, h0...; center, extra_args...)

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


# function ttest(data, h1; h0=(), center=true, max_categories=nothing)
function ttest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing, h1_missing=:skip, h0_missing=:error)
	# TODO: Filter observables with missing values (according to policy set in kwargs)

	# Something like this
	# @assert h1_missing in (:skip,:error)
	# @assert h0_missing in (:skip,:error)

	# # Filter observables with missing values if desired
	# if h1_missing == :skip || h0_missing == :skip
	#  or should this be done in covariate_stages?
	#  nah, not possible, because we need for both h0 and h1 potentially
	#  so we refactor to first get covariate_descriptions
	#  then use that to filter if desired
	#  	(NB: ensure `:` results in a no-op filter that just returns the parent datamatrix)
	#  and to create design matrix
	# end


	# Wrap single hypothesis in tuples so we can splat them below
	h0 = _splattable(h0)


	h0_covariate_descriptions, center = setup_covariate_descriptions(h0...; center)

	# TODO: We want h1 to be mean-zero (if center=true), but we don't want the intercept column. Fix.
	h1_covariate_descriptions, _ = setup_covariate_descriptions(h1; center=false)

	# Handle missing values
	skip_missing_cols = []
	let obs = get_obs_spec(data)
		if h1_missing == :skip
			col, desc = only(h1_covariate_descriptions)
			push!(skip_missing_cols, _value_vector_data_spec(obs, col, desc))
		end
		if h0_missing == :skip
			for (col,desc) in h0_covariate_descriptions
				desc === SCPCore.intercept_covariate() && continue # an intercept does not have missing values
				push!(skip_missing_cols, _value_vector_data_spec(obs, col, desc))
			end
		end
	end

	if !isempty(skip_missing_cols)
		# @show skip_missing_cols
		obs_ind = non_missing_ind_spec(skip_missing_cols...)
		data = create_datamatrix_getindex_spec(data; obs_ind) 
	end




	extra_args = max_categories === nothing ? (;) : (; max_categories)


	# We need to get the covariate model spec somewhere.
	# Because it contains the scale info needed for the ttest (for difference).

	obs = get_obs_spec(data)

	(; covariate_model_specs, covariate_specs, covariate_names) = covariate_stages(obs, h1_covariate_descriptions; center=false, extra_args...)
	h1_covariate_model = only(covariate_model_specs)
	h1_scale = covariate_scale_spec(h1_covariate_model)
	h1_design = build_designmatrix_spec(data, covariate_specs, covariate_names)

	h0_design = designmatrix_spec(data, h0...; center, extra_args...) # TODO: Use h0_covariate_descriptions and build_design_matrix?

	matrix = get_matrix(data)
	var_ids = id_column_spec(get_var_spec(data))

	ttest_table_spec(matrix, var_ids, get_matrix_spec(h1_design), h1_scale, get_matrix_spec(h0_design))
end


ttest_spec(data, h1; kwargs...) =
	create_spec(Preprocess(ttest), data, h1; kwargs...)
function Jobs.ttest(data, h1; kwargs...)
	Job(ttest_spec(data, h1; kwargs...))
end
