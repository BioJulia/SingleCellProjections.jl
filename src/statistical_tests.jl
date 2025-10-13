_splattable(x::Union{Tuple,AbstractVector}) = x
_splattable(x) = (x,)



function ftest_table_pr(action::Action, matrix, var_ids, h1_design, h0_design)
	create_spec(SCPCore.ftest_table2,
	            action(matrix), action(var_ids), action(h1_design), action(h0_design);
	            __use_cache=true, __version=v"0.0.1")
end

ftest_table_spec(matrix, var_ids, h1_design, h0_design) =
	create_spec(Projectable(ftest_table_pr), matrix, var_ids, h1_design, h0_design)


function ftest(data, h1; h0=(), center=true, max_categories=nothing)
	# TODO: Filter observables with missing values (according to policy set in kwargs)
	extra_args = max_categories === nothing ? (;) : (; max_categories)

	# Wrap single hypothesis in tuples so we can splat them below
	h1 = _splattable(h1)
	h0 = _splattable(h0)

	# Hmm. We want h1 to be mean-zero (if center=true), but we don't want the intercept column.
	h1_design = designmatrix_spec(data, h1...; center=false, extra_args...)
	h0_design = designmatrix_spec(data, h0...; center, extra_args...)

	matrix = get_matrix_spec(data)
	var_ids = create_get_ids_spec(get_var_spec(data))

	ftest_table_spec(matrix, var_ids, get_matrix_spec(h1_design), get_matrix_spec(h0_design))
end


ftest_spec(data, h1; kwargs...) =
	create_spec(Preprocess(ftest), data, h1; kwargs...)
function Jobs.ftest(data, h1; kwargs...)
	Job(ftest_spec(data, h1; kwargs...))
end





function ttest_table_pr(action::Action, matrix, var_ids, h1_design, h1_scale, h0_design)
	create_spec(SCPCore.ttest_table2,
	            action(matrix), action(var_ids), action(h1_design), prefetched(action(h1_scale)), action(h0_design);
	            __use_cache=true, __version=v"0.0.1")
end

ttest_table_spec(matrix, var_ids, h1_design, h1_scale, h0_design) =
	create_spec(Projectable(ttest_table_pr), matrix, var_ids, h1_design, h1_scale, h0_design)


function ttest(data, h1; h0=(), center=true, max_categories=nothing)
	# TODO: Filter observables with missing values (according to policy set in kwargs)
	extra_args = max_categories === nothing ? (;) : (; max_categories)

	# Wrap single hypothesis in tuples so we can splat them below
	h0 = _splattable(h0)


	# We need to get the covariate model spec somewhere.
	# Because it contains the scale info needed for the ttest (for difference).

	obs = get_obs_spec(data)
	# Hmm. We want h1 to be mean-zero (if center=true), but we don't want the intercept column.
	(; covariate_model_specs, covariate_specs, covariate_names) = covariate_stages(obs, h1; center=false, extra_args...)
	h1_covariate_model = only(covariate_model_specs)
	h1_scale = covariate_scale_spec(h1_covariate_model)
	h1_design = build_designmatrix_spec(data, covariate_specs, covariate_names)

	# Hmm. We want h1 to be mean-zero (if center=true), but we don't want the intercept column.
	h0_design = designmatrix_spec(data, h0...; center, extra_args...)

	matrix = get_matrix(data)
	var_ids = create_get_ids_spec(get_var(data))

	ttest_table_spec(matrix, var_ids, get_matrix_spec(h1_design), h1_scale, get_matrix_spec(h0_design))
end


ttest_spec(data, h1; kwargs...) =
	create_spec(Preprocess(ttest), data, h1; kwargs...)
function Jobs.ttest(data, h1; kwargs...)
	Job(ttest_spec(data, h1; kwargs...))
end
