function materialize_pseudobulk(X::SCPCore.MatrixExpression, sp)
	convert(Matrix{Float64}, X*sp)
end
materialize_pseudobulk(X, sp) = X*sp

function pseudobulk_mat(matrix, ind::AbstractVector{<:Integer}, n_combinations::Integer)
	N = size(matrix,2)
	@assert length(ind) == N
	@assert all(in(1:n_combinations), ind)
	I = 1:N

	StatsBase.counts(ind, n_combinations)
	category_weights = 1.0 ./ max.(StatsBase.counts(ind, n_combinations), 1) # avoid div by zero (but we will not even use those values below)
	weights = category_weights[ind]
	sp = sparse(I, ind, weights, N, n_combinations)
	materialize_pseudobulk(matrix, sp)
end



function pseudobulk_linear_indices(cov_ind, n_categories)
	@assert length(cov_ind) == length(n_categories)

	# Assuming we typically put sample_name as the first covariate, this leads to better cache locality when we use the sparse matrix created below.
	cov_ind = reverse(cov_ind)
	n_categories = reverse(n_categories)

	cartesian_ind = CartesianIndex.(cov_ind...)
	linear_ind = LinearIndices((n_categories...,))[cartesian_ind]
end


# TODO: This one should probably be a projectable
pseudobulk_linear_indices_spec(cov_data, cov_descs) =
	create_spec(pseudobulk_linear_indices, cov_data, cov_descs; __version=v"0.1.0")



# # TODO: handle projections
# # TODO: Throw an error if we get an unreasonably large table (e.g. larger than original)
# function repeat_categories(categories, outer, inner)
# 	repeat(categories; outer, inner)
# end
# repeat_categories_spec(categories, outer, inner) =
# 	create_spec(repeat_categories, categories, outer, inner; __version=v"0.1.1")



function pseudobulk_table_impl(::Preprocessing, categories, basenames, n_categories; extra_inner=1)
	categories = ReproducibleJobs.unsafe_unmanage(categories) # Can we avoid this?
	basenames = ReproducibleJobs.unsafe_unmanage(basenames) # Can we avoid this?
	n_categories = ReproducibleJobs.unsafe_unmanage(n_categories) # Can we avoid this?
	if categories isa ReadOnly # Can we avoid this?
		categories = categories.value
	end
	if basenames isa ReadOnly # Can we avoid this?
		basenames = basenames.value
	end
	if n_categories isa ReadOnly # Can we avoid this?
		n_categories = n_categories.value
	end

	n_cov = length(categories)
	@assert n_cov >= 1
	@assert length(basenames) == n_cov
	@assert length(n_categories) == n_cov

	# categories gives the actual values (strings), these need to be repeated
	# basenames gives column names

	outer = vcat(1, n_categories)
	outer = cumprod!(outer, outer)
	pop!(outer)

	inner = reverse!(vcat(n_categories, 1))
	cumprod!(inner, inner)
	pop!(inner)
	reverse!(inner)

	pb_cols = (name=>repeat_spec(c; outer=outer[i], inner=extra_inner*inner[i]) for (i,(name,c)) in enumerate(zip(basenames, categories)))
	create_table_spec(pb_cols...)
end

function pseudobulk_table(action::Action, categories, basenames; do_project::Bool, kwargs...)
	if do_project
		categories = action.(categories)
	end
	n_categories = fetched.(length_spec.(categories))
	create_spec(Preprocess(pseudobulk_table_impl), categories, basenames, n_categories; kwargs...)
end
pseudobulk_table_spec(categories, basenames; kwargs...) =
	create_spec(Projectable(pseudobulk_table), categories, basenames; kwargs...)



# TODO: Can we simplify this code?
function _pseudobulk_combine(a, b)
	b === nothing && return a
	a = ReproducibleJobs.unsafe_unmanage(a) # Can we avoid this?
	b = ReproducibleJobs.unsafe_unmanage(b) # Can we avoid this?
	if a isa ReadOnly # Can we avoid this?
		a = a.value
	end
	if b isa ReadOnly # Can we avoid this?
		b = b.value
	end
	vcat(a, b)
end

function pseudobulk_dm(::Mat, data, obs_cov_categories, obs_cov_ind, ::Any;
                       new_var_cov_categories = nothing,
                       new_var_cov_ind = nothing,
                       kwargs...)
	cov_categories = _pseudobulk_combine(obs_cov_categories, new_var_cov_categories)
	cov_ind = _pseudobulk_combine(obs_cov_ind, new_var_cov_ind)

	n_categories = length_spec.(cov_categories)

	linear_ind_spec = pseudobulk_linear_indices_spec(cov_ind, n_categories)

	n_combinations = prefetched(prod_spec(n_categories))
	mat = create_spec(pseudobulk_mat, get_matrix_spec(data), linear_ind_spec, n_combinations; __version=v"0.1.0")

	# reshape if needed
	if new_var_cov_categories !== nothing
		n_obs_combinations = prefetched(prod_spec(length_spec.(obs_cov_categories)))
		mat = reshape_spec(mat, :, n_obs_combinations)
	end

	cached(mat) # cache this? Or before reshape? (Probably better to cache before reshape **if** we can make it for free, e.g. by inlining specs.)
end
function pseudobulk_dm(::Var, data, args...;
                       new_var_cov_categories = nothing,
                       new_var_cov_basenames = nothing,
                       delim,
                       var_id_colname = nothing,
                       kwargs...)
	var = get_var_spec(data)
	# Standard case, we just keep the same variables
	new_var_cov_categories === nothing && return var

	# Table for the new covariates, already repeated to match number of variables
	extra_inner = fetched(table_nrow_spec(var))
	pb_table_spec = pseudobulk_table_spec(new_var_cov_categories, new_var_cov_basenames; do_project=false, extra_inner)

	# Repeat the existing var table
	n_new_var_combinations = prefetched(prod_spec(length_spec.(new_var_cov_categories)))
	var = repeat_columns_spec(var; outer=n_new_var_combinations)


	id_table = table_hcat_spec(id_column_spec(var), pb_table_spec)
	id_values = combine_column_values_spec(id_table; delim)
	var_id_colname = @something var_id_colname fetched(join_spec(get_colnames_spec(id_table), delim))

	table_hcat_spec(create_table_spec(var_id_colname=>id_values),
	                var,
	                pb_table_spec)
end
function pseudobulk_dm(::Obs, data, obs_cov_categories, ::Any, obs_cov_basenames; delim, obs_id_colname=nothing, kwargs...)
	pb_table_spec = pseudobulk_table_spec(obs_cov_categories, obs_cov_basenames; do_project=true)

	if length(obs_cov_categories)>1
		# Add ID column if we need it. Otherwise reuse the name of the single covariate.

		obs_cov_basenames = ReproducibleJobs.unsafe_unmanage(obs_cov_basenames) # Can we avoid this?
		if obs_cov_basenames isa ReadOnly # Can we avoid this?
			obs_cov_basenames = obs_cov_basenames.value
		end
		id_values = combine_column_values_spec(pb_table_spec; delim)
		obs_id_colname = @something obs_id_colname join(obs_cov_basenames, delim)
		id_spec = create_table_spec(obs_id_colname=>id_values)
		pb_table_spec = table_hcat_spec(id_spec, pb_table_spec)
	end
	pb_table_spec
end

# TODO: Get rid of this, it is needed to get some fetching to preprocess atm
function build_pseudobulk(::Preprocessing, data, args...; kwargs...)
	create_spec(DataMatrixFunction(pseudobulk_dm), data, args...; kwargs...)
end


# TODO: Consider using this to allow TwoGroup covariates too
# function setup_pseudobulk_covariates(covariates...)
# 	# Enforce all covariates to be categorical or twogroup.
# 	cov_annots = []
# 	cov_descs = []
# 	for cov in covariates
# 		if cov isa Pair
# 			let (_,desc)=cov
# 				desc isa Union{SCPCore.CategoricalCovariateDesc,SCPCore.TwoGroupCovariateDesc} || error("Pseudobulk covariates must be categorical or twogroup, got $(typeof(desc)).")
# 				# desc isa SCPCore.CategoricalCovariateDesc || error("Pseudobulk covariates must be categorical, got $(typeof(desc)).")
# 			end
# 		else
# 			cov = cov=>categorical_covariate() # default to categorical for pseudbulk
# 		end
# 		push!(cov_annots, cov.first)
# 		push!(cov_descs, cov.second)
# 	end
# 	cov_annots, cov_descs
# end

function setup_pseudobulk_covariates(covariates...)
	# Ensures all covariates are categorical.
	cov_annots = []
	for cov in covariates
		if cov isa Pair
			let (_,desc)=cov
				desc isa SCPCore.CategoricalCovariateDesc || error("Pseudobulk covariates must be categorical, got $(typeof(desc)).")
			end
			cov = cov.first
		end
		push!(cov_annots, cov)
	end
	cov_annots
end


function pseudobulk(::Preprocessing, data, obs_covariate1, obs_covariates...; new_var_covariates=nothing, delim=nothing, kwargs...)
	obs = get_obs_spec(data)

	# obs_cov_annots, obs_cov_descs = setup_pseudobulk_covariates(obs_covariate1, obs_covariates...) # Something like this for TwoGroup support
	obs_cov_annots = setup_pseudobulk_covariates(obs_covariate1, obs_covariates...)
	obs_cov_data = _extract_data_spec.(Ref(obs), obs_cov_annots)
	obs_cov_basenames = fetched.(_extract_name.(obs_cov_annots))


	# obs_cov_categories = categories_spec.(obs_cov_data, obs_cov_descs) # Something like this for TwoGroup support
	obs_cov_categories = categories_spec.(obs_cov_data)
	obs_cov_ind = indexin_spec.(obs_cov_data, obs_cov_categories)


	pb_kwargs = Pair{Symbol,Any}[]
	if new_var_covariates !== nothing
		if !(new_var_covariates isa Union{AbstractVector,Tuple})
			new_var_covariates = (new_var_covariates,) # wrap in a tuple to harmonize representation
		end

		new_var_cov_annots = setup_pseudobulk_covariates(new_var_covariates...)
		new_var_cov_data = _extract_data_spec.(Ref(obs), new_var_cov_annots)
		new_var_cov_basenames = fetched.(_extract_name.(new_var_cov_annots))

		new_var_cov_categories = categories_spec.(new_var_cov_data)
		new_var_cov_ind = indexin_spec.(new_var_cov_data, new_var_cov_categories)

		push!(pb_kwargs, :new_var_cov_categories=>new_var_cov_categories)
		push!(pb_kwargs, :new_var_cov_ind=>new_var_cov_ind)
		push!(pb_kwargs, :new_var_cov_basenames=>new_var_cov_basenames)
	end


	delim = @something delim '_'

	create_spec(Preprocess(build_pseudobulk), data, obs_cov_categories, obs_cov_ind, obs_cov_basenames; delim, pb_kwargs...)
end


pseudobulk_spec(data, obs_covariate1, obs_covariates...; kwargs...) =
	create_spec(Preprocess(pseudobulk), data, obs_covariate1, obs_covariates...; kwargs...)
function Jobs.pseudobulk(data, obs_covariate1, obs_covariates...; kwargs...)
	Job(pseudobulk_spec(data, obs_covariate1, obs_covariates...; kwargs...))
end
