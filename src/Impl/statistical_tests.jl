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
nonmissing_ind_job(column_data...) =
	create_job(nonmissing_ind, column_data...; __version=v"0.1.0")



function _filter_missing_obs(data, h::Tuple)
	@assert !isempty(h) # should be handled before calling

	# Handle missing values
	skip_missing_cols = []
	obs = SCP.get_obs(data)

	for a in h
		if a isa Pair
			a = a.first
		end
		push!(skip_missing_cols, _extract_data_job(obs, a))
	end

	skip_missing_cols = (_extract_data_job(obs, a isa Pair ? a.first : a) for a in h)
	obs_ind = nonmissing_ind_job(skip_missing_cols...)
	create_datamatrix_getindex_job(data; obs_ind)
end

function _filter_missing_obs(data; h1::Union{Tuple,AbstractVector}, h1_missing,
                                   h0::Union{Tuple,AbstractVector}, h0_missing)
	h1b = h1_missing == :skip ? h1 : ()
	h0b = h0_missing == :skip ? h0 : ()
	h = (h1b..., h0b...)
	isempty(h) ? data : _filter_missing_obs(data, h)
end






function ftest_table_pr(action::Action, matrix, var, h1_design, h0_design; do_sort)
	cached(create_job(SCPCore.ftest_table,
	                  action(matrix), action(var), action(h1_design), action(h0_design);
	                  do_sort,
	                  __version=v"0.0.1"))
end

ftest_table_job(matrix, var, h1_design, h0_design; kwargs...) =
	create_job(Projectable(ftest_table_pr), matrix, var, h1_design, h0_design; kwargs...)


function ftest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing, h1_missing=:skip, h0_missing=:error, var_cols=nothing, do_sort=true)
	@assert h1_missing in (:skip,:error)
	@assert h0_missing in (:skip,:error)

	# Wrap single hypothesis in tuples so we can splat them below
	h1 = _splattable(h1)
	h0 = _splattable(h0)

	# Handle missing values
	data = _filter_missing_obs(data; h1, h0, h1_missing, h0_missing)

	extra_kwargs = max_categories === nothing ? (;) : (; max_categories)

	# Hmm. We want h1 to be mean-zero (if center=true), but we don't want the intercept column.
	h1_design = SCP.designmatrix(data, h1...; center=false, extra_kwargs...)
	h0_design = SCP.designmatrix(data, h0...; center, extra_kwargs...)

	matrix = SCP.get_matrix(data)

	var = SCP.get_var(data)
	table_var = SCP.id_column(var)
	if var_cols !== nothing
		var_cols = _splattable(var_cols)
		table_var = SCP.table_hcat(table_var, SCP.get_columns(var, var_cols...))
	end

	ftest_table_job(matrix, table_var, SCP.get_matrix(h1_design), SCP.get_matrix(h0_design); do_sort)
end







function ttest_table_pr(action::Action, matrix, var, h1_design, h1_scale, h0_design; do_sort)
	cached(create_job(SCPCore.ttest_table,
	                  action(matrix), action(var),
	                  action(h1_design), prefetched(action(h1_scale)),
	                  action(h0_design);
	                  do_sort,
	                  __version=v"0.0.1"))
end

ttest_table_job(matrix, var, h1_design, h1_scale, h0_design; kwargs...) =
	create_job(Projectable(ttest_table_pr), matrix, var, h1_design, h1_scale, h0_design; kwargs...)



# TODO: This does not work properly with projections. Fix.
function ttest(::Preprocessing, data, h1; h0=(), center=true, max_categories=nothing, h1_missing=:skip, h0_missing=:error, var_cols=nothing, do_sort=true)
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
		h1 = h1=>SCP.numerical_covariate() # default to numerical if not given - we want something 1d
	end


	# Handle missing values
	data = _filter_missing_obs(data; h1=_splattable(h1), h0, h1_missing, h0_missing)

	obs = SCP.get_obs(data)


	extra_kwargs = max_categories === nothing ? (;) : (; max_categories)


	h1_cov_annot, h1_cov_desc = h1


	center = center || (h1_cov_desc isa TwoGroupCovariateDesc) # Center if h1 requires it
	if !center # Figure out if h0 requires centering
		_, h0_cov_descs = setup_covariate_descriptions(obs, h0...)
		center = fetched(has_centering_job(h0_cov_descs))
	end


	h0_design = SCP.designmatrix(data, h0...; center, extra_kwargs...)

	h1_cov_data = _extract_data_job(obs, h1_cov_annot)
	ms = mean_and_scale_job(h1_cov_data, h1_cov_desc; center)
	h1_scale = fetched(getindex_job(ms, 2))
	h1_design_mat = covariate_matrix_job(h1_cov_data, h1_cov_desc; center) # center affects this column, but we don't get an intercept

	matrix = SCP.get_matrix(data)

	var = SCP.get_var(data)
	table_var = SCP.id_column(var)
	if var_cols !== nothing
		var_cols = _splattable(var_cols)
		table_var = SCP.table_hcat(table_var, SCP.get_columns(var, var_cols...))
	end

	ttest_table_job(matrix, table_var, h1_design_mat, h1_scale, SCP.get_matrix(h0_design); do_sort)
end




# groups vector - Projectable so projection recomputes it on the projected obs column, using the frozen group labels.
# `nothing` (the "group A vs the rest" case) must not be passed as a spec argument, so the group labels are
# splatted in (see `mannwhitney_groups_job`) - matching how `_group_args` strips a `nothing` group_b in design.jl.
mannwhitney_groups_pr(action::Action, cov_data, group_labels...; h1_missing) =
	create_job(SCPCore.mannwhitney_groups, action(cov_data), group_labels...; h1_missing, __version=v"0.1.0")

mannwhitney_groups_job(cov_data, group_labels...; kwargs...) =
	create_job(Projectable(mannwhitney_groups_pr), cov_data, group_labels...; kwargs...)


mannwhitney_table_pr(action::Action, matrix, var, groups; kwargs...) =
	cached(create_job(SCPCore.mannwhitney_table, action(matrix), action(var), action(groups); kwargs..., __version=v"0.0.2"))

mannwhitney_table_job(matrix, var, groups; kwargs...) =
	create_job(Projectable(mannwhitney_table_pr), matrix, var, groups; kwargs...)


function mannwhitney(::Preprocessing, data, column, group_a=nothing, group_b=nothing;
                     h1_missing=:skip, do_sort=true, kwargs...)
	@assert h1_missing in (:skip,:error)

	obs = SCP.get_obs(data)
	cov_data = _extract_data_job(obs, column)

	if group_a === nothing
		# Auto-detect the two labels from the base data and freeze them: they are bound to the base
		# `cov_data` and never `action`-projected, so a projected test reuses them instead of re-inferring.
		@assert group_b === nothing
		resolved = create_job(SCPCore.mannwhitney_resolve_groups, cov_data; __version=v"0.1.0")
		groups = mannwhitney_groups_job(cov_data, fetched(getindex_job(resolved, 1)), fetched(getindex_job(resolved, 2)); h1_missing)
	elseif group_b === nothing
		# Group A vs the rest - leave group_b unset (never pass `nothing` into a spec).
		groups = mannwhitney_groups_job(cov_data, group_a; h1_missing)
	else
		groups = mannwhitney_groups_job(cov_data, group_a, group_b; h1_missing)
	end

	matrix = SCP.get_matrix(data)
	var = SCP.id_column(SCP.get_var(data))

	mannwhitney_table_job(matrix, var, groups; do_sort, kwargs...)
end



