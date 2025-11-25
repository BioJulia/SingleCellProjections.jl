# No, we want to implement this at the column level. It's just a repeat with inner and outer. If we know the number of elements in each category.
# function cartesian_product_of_categories(df::DataFrame)
# 	u = (DataFrame(name=>unique(values)) for (name=>values) in pairs(eachcol(df)))
# 	res = crossjoin(u...)
# end

# TODO: Throw an error if we get an unreasonably large table (e.g. larger than original)
function repeat_categories(uv, nvalues, ind)
	@assert nvalues[ind] == length(uv)
	repeat(uv; outer=prod(nvalues[1:ind-1]), inner=prod(nvalues[ind+1:end]))
end
repeat_categories_spec(uv, nvalues, ind) =
	create_spec(repeat_categories, uv, nvalues, ind; __version=v"0.1.0")


function cartesian_product_of_categories_fallback(table::DataFrame)
	u = [unique(v) for v in eachcol(table)] # unique values in each column
	n = length.(u)
	# create a new table, with the original column names, but with inner/outer repeats depending the number of unique values in preceeding/suceeding columns
	cols = (k=>repeat_categories(uv, n, i) for (i,(k,uv)) in enumerate(zip(names(table), u)))
	DataFrame(cols...)
end
function cartesian_product_of_categories(::Preprocessing{E}, table) where E
	if is_create_table(table)
		u = [unique_spec(v) for (k,v) in table.args] # unique values in each column
		n = prefetched.(length_spec.(u))
		# create a new table, with the original column names, but with inner/outer repeats depending the number of unique values in preceeding/suceeding columns
		cols = (k=>repeat_categories_spec(uv, n, i) for (i,((k,_),uv)) in enumerate(zip(table.args, u)))
		create_table_spec(cols...)
	elseif E
		create_spec(Preprocess{false}(cartesian_product_of_categories), table)
	else
		create_spec(cartesian_product_of_categories_fallback, table; __version=v"0.1.2")
	end
end
cartesian_product_of_categories_spec(table) = create_spec(Preprocess(cartesian_product_of_categories), table)


function pseudobulk_id(::Preprocessing, table; id_colname=nothing, delim='_')
	if id_colname === nothing
		cn = get_colnames_spec(table)
		id_colname = fetched(join_spec(cn, delim))
		pseudobulk_id_spec(table; id_colname, delim) # Preprocess again - with fetched id_colname
	else
		unique_combinations = cartesian_product_of_categories_spec(table)
		id_values = combine_column_values_spec(unique_combinations; delim)
		create_table_spec(id_colname=>id_values)
	end
end
pseudobulk_id_spec(table; kwargs...) = create_spec(Preprocess(pseudobulk_id), table; kwargs...)



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




function pseudobulk(f::Union{Mat,Obs}, data, colnames...; delim='_')
	cols = get_columns_spec(get_obs_spec(data), colnames...)
	pb_id = pseudobulk_id_spec(cols; delim)

	if f isa Obs
		pb_id # TODO: Add more annotations?
	else#if f === Mat
		id_per_obs = combine_column_values_spec(cols; delim)
		ind = indexin_spec(id_per_obs, column_data_spec(pb_id,1); not_found=:error) # indices matching each original obs to a pseudobulk obs
		create_spec(pseudobulk_mat, get_matrix_spec(data), ind, table_nrow_spec(pb_id); __version=v"0.1.1")
	end
end
pseudobulk(::Var, data, args...; kwargs...) = get_var_spec(data)



pseudobulk_spec(data, colname1, colnames...; kwargs...) =
	create_spec(DataMatrixFunction(pseudobulk), data, colname1, colnames...; kwargs...)
function Jobs.pseudobulk(data, colname1, colnames...; kwargs...)
	Job(pseudobulk_spec(data, colname1, colnames...; kwargs...))
end



