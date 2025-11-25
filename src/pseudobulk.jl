# TODO: Throw an error if we get an unreasonably large table (e.g. larger than original)
function repeat_categories(uv, nvalues, ind)
	@assert nvalues[ind] == length(uv)
	repeat(uv; outer=prod(nvalues[1:ind-1]), inner=prod(nvalues[ind+1:end]))
end
repeat_categories_spec(uv, nvalues, ind) =
	create_spec(repeat_categories, uv, nvalues, ind; __version=v"0.1.0")




# NB: It is assumed below that this creates a new vector (that the caller is allowed to modify)
function unique_column_values_specs(table, colnames)
	if colnames isa ReadOnly # Can we avoid this?
		colnames = colnames.value
	end
	[unique_spec(column_data_spec(table, cn)) for cn in colnames]
end




function cartesian_product_of_categories(::Preprocessing, colnames, uv::AbstractVector)
	if colnames isa ReadOnly # Can we avoid this?
		colnames = colnames.value
	end

	@assert length(colnames) == length(uv)
	n = prefetched.(length_spec.(uv))
	cols = (name=>repeat_categories_spec(x, n, i) for (i,(name,x)) in enumerate(zip(colnames, uv)))
	create_table_spec(cols...)
end
cartesian_product_of_categories_spec(colnames, uv::AbstractVector) = create_spec(Preprocess(cartesian_product_of_categories), colnames, uv)



function pseudobulk_id_values(::Preprocessing, colnames, unique_values; delim='_')
	# unique_values = unique_column_values_specs(table, colnames)
	unique_combinations = cartesian_product_of_categories_spec(colnames, unique_values)
	id_values = combine_column_values_spec(unique_combinations; delim)
end
pseudobulk_id_values_spec(colnames, unique_values; kwargs...) = create_spec(Preprocess(pseudobulk_id_values), colnames, unique_values; kwargs...)


pseudobulk_var_id_colname(colnames, id_colname; delim) = join(vcat(colnames, id_colname), delim)



function materialize_pseudobulk(X::SCPCore.MatrixExpression, sp)
	convert(Matrix{Float64}, X*sp)
end
materialize_pseudobulk(X, sp) = X*sp

# TODO: decide how projections are handled
function pseudobulk_mat(matrix, ind::AbstractVector{<:Integer}, n_categories)
	N = size(matrix,2)
	@assert length(ind) == N
	@assert all(in(1:n_categories), ind)
	I = 1:N

	StatsBase.counts(ind, n_categories)
	category_weights = 1.0 ./ max.(StatsBase.counts(ind, n_categories), 1) # avoid div by zero (but we will not even use those values below)
	weights = category_weights[ind]
	sp = sparse(I, ind, weights, N, n_categories)
	materialize_pseudobulk(matrix, sp)
end




function pseudobulk(::Mat, data, colnames...; delim='_', new_var_colnames=(), kwargs...)
	colnames = collect(colnames)
	new_var_colnames = collect(new_var_colnames)

	obs = get_obs_spec(data)
	all_colnames = vcat(colnames, new_var_colnames) # order here matters for reshape to work

	unique_values = unique_column_values_specs(obs, all_colnames)
	# pb_id_values = pseudobulk_id_values_spec(obs, all_colnames; delim)
	pb_id_values = pseudobulk_id_values_spec(all_colnames, unique_values; delim)
	id_per_obs = combine_column_values_spec(get_columns_spec(obs, all_colnames...); delim)
	ind = indexin_spec(id_per_obs, pb_id_values; not_found=:error) # indices matching each original obs to a pseudobulk obs
	mat = create_spec(pseudobulk_mat, get_matrix_spec(data), ind, length_spec(pb_id_values); __version=v"0.1.1")

	if !isempty(new_var_colnames)
		# We need to reshape the matrix, because we are creating new variables

		# we need the number of vars, and the product of the length of the new var columns
		# or just the product of the lengths of the obs columns
		new_obs_unique_values = unique_column_values_specs(obs, colnames)
		n_new_obs = prefetched(prod_spec(length_spec.(new_obs_unique_values)))
		mat = reshape_spec(mat, :, n_new_obs)
	end
	cached(mat) # or should we cache before the reshape?
end
function pseudobulk(::Var, data, args...; delim='_', new_var_colnames=(), new_var_id_colname=nothing, kwargs...)
	if !isempty(new_var_colnames)
		new_var_colnames = collect(new_var_colnames)
		var = get_var_spec(data)
		obs = get_obs_spec(data)
		unique_values = unique_column_values_specs(obs, new_var_colnames)
		push!(unique_values, id_column_data_spec(var)) # unique_column_values_specs creates a fresh vector, so we are allowed to push to it.
		colnames = vcat(new_var_colnames, get_id_colname_spec(var))

		# Hmm. If we create the table first, we can just get the new IDs by joining columns.
		new_var_id_values = pseudobulk_id_values_spec(colnames, unique_values; delim)
		new_var_id_colname = create_spec(pseudobulk_var_id_colname, new_var_colnames, get_id_colname_spec(var); delim, __version=v"0.1.0")

		# First we want "pb_id" column.
		# Then we want the new_var_colnames repeated (annotations from obs lifted over to var)
		# Then we want all var columns repeated.
		# How do we handle name clashes?

		create_table_spec(new_var_id_colname=>new_var_id_values)
	else
		@assert new_var_id_colname===nothing # Only allow renaming if new variables are created
		get_var_spec(data)
	end
end
function pseudobulk(::Obs, data, colnames...; delim='_', id_colname=nothing, kwargs...)
	colnames = collect(colnames)
	obs = get_obs_spec(data)

	unique_values = unique_column_values_specs(obs, colnames)
	pb_id_values = pseudobulk_id_values_spec(colnames, unique_values; delim)

	id_colname = @something id_colname join(colnames, delim)
	create_table_spec(id_colname=>pb_id_values)
end


# TODO: Use covariates so we can provide external annotations too?
pseudobulk_spec(data, colname1, colnames...; kwargs...) =
	create_spec(DataMatrixFunction(pseudobulk), data, colname1, colnames...; kwargs...)
function Jobs.pseudobulk(data, colname1, colnames...; kwargs...)
	Job(pseudobulk_spec(data, colname1, colnames...; kwargs...))
end



