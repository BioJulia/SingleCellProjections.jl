function materialize_pseudobulk(X::SCPCore.MatrixExpression, sp)
	convert(Matrix{Float64}, X*sp)
end
materialize_pseudobulk(X, sp) = X*sp

function pseudobulk_mat(matrix, (ind,n_combinations)::Tuple{<:AbstractVector{<:Integer},<:Integer})
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



function _verify_models(vv_models)
	bad_ind = findall(x->!(x isa Union{CategoricalValueVectorModel,TwoGroupValueVectorModel}), vv_models)
	if any(bad_ind)
		bad_models = unique(typeof.(vv_models[bad_ind]))
		throw(ArgumentError("Only Categorical and TwoGroup covariates are allowed in pseudobulk, found $bad_models."))
	end
end


function pseudobulk_linear_indices(vv_models, value_vectors)
	_verify_models(vv_models)

	# Assuming we typically put sample_name as the first covariate, this leads to better cache locality when we use the sparse matrix created below.
	vv_models = reverse(vv_models)
	value_vectors = reverse(value_vectors)


	multi_ind = Vector{Int}[vv.values for vv in value_vectors]
	n_categories = Int[get_n_categories(model) for model in vv_models]

	cartesian_ind = CartesianIndex.(multi_ind...)
	linear_ind = LinearIndices((n_categories...,))[cartesian_ind]

	n_combinations = prod(n_categories) # i.e. linear_ind are in 1:n_combinations
	linear_ind, n_combinations
end


# TODO: This one should probably be a projectable
pseudobulk_linear_indices_spec(vv_model_specs, vv_specs) =
	create_spec(pseudobulk_linear_indices, vv_model_specs, vv_specs; __version=v"0.1.0")



# # TODO: handle projections
# # TODO: Throw an error if we get an unreasonably large table (e.g. larger than original)
# function repeat_categories(categories, nvalues, ind)
# 	@assert nvalues[ind] == length(categories)
# 	repeat(categories; outer=prod(nvalues[1:ind-1]), inner=prod(nvalues[ind+1:end]))
# end
# repeat_categories(m::CategoricalValueVectorModel, nvalues, ind) = repeat_categories(m.categories, nvalues, ind)

# repeat_categories_spec(categories, nvalues, ind) =
# 	create_spec(repeat_categories, categories, nvalues, ind; __version=v"0.1.0")


# TODO: handle projections
# TODO: Throw an error if we get an unreasonably large table (e.g. larger than original)
function repeat_categories(categories, outer, inner)
	repeat(categories; outer, inner)
end
repeat_categories(m::CategoricalValueVectorModel, outer, inner) = repeat_categories(m.categories, outer, inner)

repeat_categories_spec(categories, outer, inner) =
	create_spec(repeat_categories, categories, outer, inner; __version=v"0.1.1")



function pseudobulk_table_impl(::Preprocessing, vv_model_specs, base_names, n_categories)
	vv_model_specs = ReproducibleJobs.unsafe_unmanage(vv_model_specs)
	base_names = ReproducibleJobs.unsafe_unmanage(base_names)
	if base_names isa ReadOnly # Can we avoid this?
		base_names = base_names.value
	end
	if n_categories isa ReadOnly # Can we avoid this?
		n_categories = n_categories.value
	end

	n_obs_cov = length(vv_model_specs)
	@assert n_obs_cov >= 1
	@assert length(base_names) == n_obs_cov
	@assert length(n_categories) == n_obs_cov

	# vv_model gives the actual values (strings), these need to be repeated
	# base_names gives column names

	outer = vcat(1, n_categories)
	outer = cumprod!(outer, outer)
	pop!(outer)

	inner = reverse!(vcat(n_categories, 1))
	cumprod!(inner, inner)
	pop!(inner)
	reverse!(inner)

	pb_cols = (prefetched(name)=>repeat_categories_spec(m, outer[i], inner[i]) for (i,(name,m)) in enumerate(zip(base_names, vv_model_specs)))
	create_table_spec(pb_cols...)
end

function pseudobulk_table(action::Action, vv_model_specs, base_name_specs; do_project::Bool)
	if do_project
		vv_model_specs = action.(vv_model_specs)
		base_name_specs = action.(base_name_specs)
	end
	n_categories = fetched.(get_n_categories_spec.(vv_model_specs))	
	create_spec(Preprocess(pseudobulk_table_impl), vv_model_specs, fetched.(base_name_specs), n_categories)
end
pseudobulk_table_spec(vv_model_specs, base_name_specs; kwargs...) =
	create_spec(Projectable(pseudobulk_table), vv_model_specs, base_name_specs; kwargs...)



function pseudobulk_dm(::Mat, data, vv_model_specs, vv_specs, base_name_specs; kwargs...)
	# TODO: Support new_var_covariates

	# The vv_specs (and vv_model_specs) define multidimensional indices.
	# We want to convert that into linear indices
	linear_ind_spec = pseudobulk_linear_indices_spec(vv_model_specs, vv_specs)
	mat = create_spec(pseudobulk_mat, get_matrix_spec(data), linear_ind_spec; __version=v"0.1.0")

	cached(mat) # cache this? Or before reshape? (Probably better to cache before reshape **if** we can make it for free, e.g. by inlining specs.)
end
function pseudobulk_dm(::Var, data, args...; kwargs...)
	# TODO: Support new_var_covariates
	get_var_spec(data)
end
function pseudobulk_dm(::Obs, data, vv_model_specs, vv_specs, base_name_specs; delim, id_colname=nothing)
	pb_table_spec = pseudobulk_table_spec(vv_model_specs, base_name_specs; do_project=true)
	id_values = combine_column_values_spec(pb_table_spec; delim)
	# id_colname = @something id_colname join(colnames, delim) # Needs some preprocessing
	id_colname = "pb_id" # TEMP
	id_spec = create_table_spec(id_colname=>id_values)
	table_hcat_spec(id_spec, pb_table_spec)
end


function pseudobulk(::Preprocessing, data, obs_covariate1, obs_covariates...; new_var_covariates=nothing, delim=nothing, kwargs...)
	obs = get_obs_spec(data)
	covariate_descriptions, center = setup_covariate_descriptions(obs_covariate1, obs_covariates...; center=false)
	# TODO: covariate_stages creates more specs that are discarded here, avoid that by refactoring covariate_stages somehow.
	(; vv_model_specs, vv_specs, base_name_specs) = covariate_stages(obs, covariate_descriptions; center, kwargs...)

	# TODO: Where do we ensure that only Categorical (and TwoGroup one vs all) covariates are specified?
	# I guess we can only know later, due to auto_covariate()

	pb_kwargs = Pair{Symbol,Any}[]
	new_var_covariates !== nothing && push!(pb_kwargs, :new_var_covariates=>new_var_covariates)
	delim = @something delim '_'

	create_spec(DataMatrixFunction(pseudobulk_dm), data, vv_model_specs, vv_specs, base_name_specs; delim, pb_kwargs...)
end


pseudobulk_spec(data, obs_covariate1, obs_covariates...; kwargs...) =
	create_spec(Preprocess(pseudobulk), data, obs_covariate1, obs_covariates...; kwargs...)
function Jobs.pseudobulk(data, obs_covariate1, obs_covariates...; kwargs...)
	Job(pseudobulk_spec(data, obs_covariate1, obs_covariates...; kwargs...))
end




# Old

# # TODO: Throw an error if we get an unreasonably large table (e.g. larger than original)
# function repeat_categories(uv, nvalues, ind)
# 	@assert nvalues[ind] == length(uv)
# 	repeat(uv; outer=prod(nvalues[1:ind-1]), inner=prod(nvalues[ind+1:end]))
# end
# repeat_categories_spec(uv, nvalues, ind) =
# 	create_spec(repeat_categories, uv, nvalues, ind; __version=v"0.1.0")




# # NB: It is assumed below that this creates a new vector (that the caller is allowed to modify)
# function unique_column_values_specs(table, colnames)
# 	if colnames isa ReadOnly # Can we avoid this?
# 		colnames = colnames.value
# 	end
# 	[unique_spec(column_data_spec(table, cn)) for cn in colnames]
# end




# function cartesian_product_of_categories(::Preprocessing, colnames, uv::AbstractVector)
# 	if colnames isa ReadOnly # Can we avoid this?
# 		colnames = colnames.value
# 	end

# 	@assert length(colnames) == length(uv)
# 	n = prefetched.(length_spec.(uv))
# 	cols = (name=>repeat_categories_spec(x, n, i) for (i,(name,x)) in enumerate(zip(colnames, uv)))
# 	create_table_spec(cols...)
# end
# cartesian_product_of_categories_spec(colnames, uv::AbstractVector) = create_spec(Preprocess(cartesian_product_of_categories), colnames, uv)



# function pseudobulk_id_values(::Preprocessing, colnames, unique_values; delim='_')
# 	# unique_values = unique_column_values_specs(table, colnames)
# 	unique_combinations = cartesian_product_of_categories_spec(colnames, unique_values)
# 	id_values = combine_column_values_spec(unique_combinations; delim)
# end
# pseudobulk_id_values_spec(colnames, unique_values; kwargs...) = create_spec(Preprocess(pseudobulk_id_values), colnames, unique_values; kwargs...)


# pseudobulk_var_id_colname(colnames, id_colname; delim) = join(vcat(colnames, id_colname), delim)



# function materialize_pseudobulk(X::SCPCore.MatrixExpression, sp)
# 	convert(Matrix{Float64}, X*sp)
# end
# materialize_pseudobulk(X, sp) = X*sp

# # TODO: decide how projections are handled
# function pseudobulk_mat(matrix, ind::AbstractVector{<:Integer}, n_categories)
# 	N = size(matrix,2)
# 	@assert length(ind) == N
# 	@assert all(in(1:n_categories), ind)
# 	I = 1:N

# 	StatsBase.counts(ind, n_categories)
# 	category_weights = 1.0 ./ max.(StatsBase.counts(ind, n_categories), 1) # avoid div by zero (but we will not even use those values below)
# 	weights = category_weights[ind]
# 	sp = sparse(I, ind, weights, N, n_categories)
# 	materialize_pseudobulk(matrix, sp)
# end




# function pseudobulk(::Mat, data, colnames...; delim='_', new_var_colnames=(), kwargs...)
# 	colnames = collect(colnames)
# 	new_var_colnames = collect(new_var_colnames)

# 	obs = get_obs_spec(data)
# 	all_colnames = vcat(colnames, new_var_colnames) # order here matters for reshape to work

# 	unique_values = unique_column_values_specs(obs, all_colnames)
# 	# pb_id_values = pseudobulk_id_values_spec(obs, all_colnames; delim)
# 	pb_id_values = pseudobulk_id_values_spec(all_colnames, unique_values; delim)
# 	id_per_obs = combine_column_values_spec(get_columns_spec(obs, all_colnames...); delim)
# 	ind = indexin_spec(id_per_obs, pb_id_values; not_found=:error) # indices matching each original obs to a pseudobulk obs
# 	mat = create_spec(pseudobulk_mat, get_matrix_spec(data), ind, length_spec(pb_id_values); __version=v"0.1.1")

# 	if !isempty(new_var_colnames)
# 		# We need to reshape the matrix, because we are creating new variables

# 		# we need the number of vars, and the product of the length of the new var columns
# 		# or just the product of the lengths of the obs columns
# 		new_obs_unique_values = unique_column_values_specs(obs, colnames)
# 		n_new_obs = prefetched(prod_spec(length_spec.(new_obs_unique_values)))
# 		mat = reshape_spec(mat, :, n_new_obs)
# 	end
# 	cached(mat) # or should we cache before the reshape?
# end
# function pseudobulk(::Var, data, args...; delim='_', new_var_colnames=(), new_var_id_colname=nothing, kwargs...)
# 	if !isempty(new_var_colnames)
# 		new_var_colnames = collect(new_var_colnames)
# 		var = get_var_spec(data)
# 		obs = get_obs_spec(data)
# 		unique_values = unique_column_values_specs(obs, new_var_colnames)


# 		# # TODO: Remove this code
# 		# # Hmm. If we create the table first, we can just get the new IDs by joining columns.
# 		# push!(unique_values, id_column_data_spec(var)) # unique_column_values_specs creates a fresh vector, so we are allowed to push to it.
# 		# colnames = vcat(new_var_colnames, get_id_colname_spec(var))
# 		# new_var_id_values = pseudobulk_id_values_spec(colnames, unique_values; delim)
# 		# new_var_id_colname = create_spec(pseudobulk_var_id_colname, new_var_colnames, get_id_colname_spec(var); delim, __version=v"0.1.0")

# 		# First we want "pb_id" column.
# 		# Then we want the new_var_colnames repeated (annotations from obs lifted over to var) (inner=...)
# 		# Then we want all var columns repeated. (outer=...)
# 		# How do we handle name clashes?


# 		# And replace with this
# 		unique_combinations = cartesian_product_of_categories_spec(new_var_colnames, unique_values)
# 		# repeat the table nvar times

# 		# and repeat the var table to match
# 		# merge them
# 		# and insert id_column first



# 		create_table_spec(new_var_id_colname=>new_var_id_values)
# 	else
# 		@assert new_var_id_colname===nothing # Only allow renaming if new variables are created
# 		get_var_spec(data)
# 	end
# end
# function pseudobulk(::Obs, data, colnames...; delim='_', id_colname=nothing, kwargs...)
# 	colnames = collect(colnames)
# 	obs = get_obs_spec(data)

# 	unique_values = unique_column_values_specs(obs, colnames)
# 	pb_id_values = pseudobulk_id_values_spec(colnames, unique_values; delim)

# 	id_colname = @something id_colname join(colnames, delim)
# 	create_table_spec(id_colname=>pb_id_values)
# end


# # TODO: Use covariates so we can provide external annotations too?
# function pseudobulk_spec(data, colname1, colnames...; new_var_colnames=nothing, kwargs...)
# 	# Ensure new_var_colnames is not present - or wrapped in a vector/tuple. (This harmonizes the representation, allowing e.g. a string to be passed.)
# 	if new_var_colnames === nothing
# 		extra_kwargs = (;)
# 	elseif new_var_colnames isa Union{AbstractVector,Tuple}
# 		extra_kwargs = (; new_var_colnames)
# 	else
# 		extra_kwargs = (; new_var_colnames=(new_var_colnames,))
# 	end

# 	create_spec(DataMatrixFunction(pseudobulk), data, colname1, colnames...; extra_kwargs..., kwargs...)
# end
# function Jobs.pseudobulk(data, colname1, colnames...; kwargs...)
# 	Job(pseudobulk_spec(data, colname1, colnames...; kwargs...))
# end



