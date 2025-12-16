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



function _filter_missing_obs(data, h::Tuple)
	@assert !isempty(h) # should be handled before calling

	# Handle missing values
	skip_missing_cols = []
	obs = get_obs_spec(data)

	for a in h
		if a isa Pair
			a = a.first
		end
		push!(skip_missing_cols, _extract_data_spec(obs, a))
	end

	skip_missing_cols = (_extract_data_spec(obs, a isa Pair ? a.first : a) for a in h)
	obs_ind = nonmissing_ind_spec(skip_missing_cols...)
	create_datamatrix_getindex_spec(data; obs_ind)
end

function _filter_missing_obs(data; h1::Union{Tuple,AbstractVector}, h1_missing,
                                   h0::Union{Tuple,AbstractVector}, h0_missing)
	h1b = h1_missing == :skip ? h1 : ()
	h0b = h0_missing == :skip ? h0 : ()
	h = (h1b..., h0b...)
	isempty(h) ? data : _filter_missing_obs(data, h)
end






function ftest_table_pr(action::Action, matrix, var, h1_design, h0_design)
	cached(create_spec(SCPCore.ftest_table2,
	                   action(matrix), action(var), action(h1_design), action(h0_design);
	                   __version=v"0.0.1"))
end

ftest_table_spec(matrix, var, h1_design, h0_design) =
	create_spec(Projectable(ftest_table_pr), matrix, var, h1_design, h0_design)


function ftest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing, h1_missing=:skip, h0_missing=:error, var_cols=nothing)
	@assert h1_missing in (:skip,:error)
	@assert h0_missing in (:skip,:error)

	# Wrap single hypothesis in tuples so we can splat them below
	h1 = _splattable(h1)
	h0 = _splattable(h0)

	# Handle missing values
	data = _filter_missing_obs(data; h1, h0, h1_missing, h0_missing)

	extra_kwargs = max_categories === nothing ? (;) : (; max_categories)

	# Hmm. We want h1 to be mean-zero (if center=true), but we don't want the intercept column.
	h1_design = designmatrix_spec(data, h1...; center=false, extra_kwargs...)
	h0_design = designmatrix_spec(data, h0...; center, extra_kwargs...)

	matrix = get_matrix_spec(data)

	var = get_var_spec(data)
	table_var = id_column_spec(var)
	if var_cols !== nothing
		var_cols = _splattable(var_cols)
		table_var = table_hcat_spec(table_var, get_columns_spec(var, var_cols...))
	end

	ftest_table_spec(matrix, table_var, get_matrix_spec(h1_design), get_matrix_spec(h0_design))
end


ftest_spec(data, h1; kwargs...) =
	create_spec(Preprocess(ftest), data, h1; kwargs...)
function Jobs.ftest(data, h1; kwargs...)
	Job(ftest_spec(data, h1; kwargs...))
end





function ttest_table_pr(action::Action, matrix, var, h1_design, h1_scale, h0_design)
	cached(create_spec(SCPCore.ttest_table2,
	                   action(matrix), action(var),
	                   action(h1_design), prefetched(action(h1_scale)),
	                   action(h0_design);
	                   __version=v"0.0.1"))
end

ttest_table_spec(matrix, var, h1_design, h1_scale, h0_design) =
	create_spec(Projectable(ttest_table_pr), matrix, var, h1_design, h1_scale, h0_design)



# TODO: This does not work properly with projections. Fix.
function ttest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing, h1_missing=:skip, h0_missing=:error, var_cols=nothing)
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
	data = _filter_missing_obs(data; h1=_splattable(h1), h0, h1_missing, h0_missing)

	obs = get_obs_spec(data)


	extra_kwargs = max_categories === nothing ? (;) : (; max_categories)


	h1_cov_annot, h1_cov_desc = h1


	center = center || (h1_cov_desc isa TwoGroupCovariateDesc) # Center if h1 requires it
	if !center # Figure out if h0 requires centering
		_, h0_cov_descs = setup_covariate_descriptions(obs, h0...)
		center = fetched(has_centering_spec(h0_cov_descs))
	end


	h0_design = designmatrix_spec(data, h0...; center, extra_kwargs...)

	h1_cov_data = _extract_data_spec(obs, h1_cov_annot)
	ms = mean_and_scale_spec(h1_cov_data, h1_cov_desc; center)
	h1_scale = fetched(getindex_spec(ms, 2))
	h1_design_mat = covariate_matrix_spec(h1_cov_data, h1_cov_desc; center) # center affects this column, but we don't get an intercept

	matrix = get_matrix_spec(data)

	var = get_var_spec(data)
	table_var = id_column_spec(var)
	if var_cols !== nothing
		var_cols = _splattable(var_cols)
		table_var = table_hcat_spec(table_var, get_columns_spec(var, var_cols...))
	end

	ttest_table_spec(matrix, table_var, h1_design_mat, h1_scale, get_matrix_spec(h0_design))
end



ttest_spec(data, h1; kwargs...) =
	create_spec(Preprocess(ttest), data, h1; kwargs...)
function Jobs.ttest(data, h1; kwargs...)
	Job(ttest_spec(data, h1; kwargs...))
end
