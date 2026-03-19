annotate(::Mat, data; kwargs...) = get_matrix_spec(data)
function annotate(f::Union{Var,Obs}, data; kwargs...)
	s = get_spec(f, data)
	df = get(kwargs, f isa Var ? :var : :obs, nothing)
	df === nothing && return s
	table_leftjoin_spec(s, df)
end

Jobs.annotate_obs(data, df; kwargs...) =
	create_spec(DataMatrixFunction(annotate), data; kwargs..., obs=df)
Jobs.annotate_var(data, df; kwargs...) =
	create_spec(DataMatrixFunction(annotate), data; kwargs..., var=df)
Jobs.annotate(data, var_df, obs_df; kwargs...) =
	create_spec(DataMatrixFunction(annotate), data; kwargs..., var=var_df, obs=obs_df)



add_var_column(f::Union{Mat,Obs}, data, name, column) = get_spec(f, data)
add_var_column(::Var, data, name, column) = add_column_spec(get_var_spec(data), name, column)
Jobs.add_var_column(data, name, column) =
	create_spec(DataMatrixFunction(add_var_column), data, name, column)

add_obs_column(f::Union{Mat,Var}, data, name, column) = get_spec(f, data)
add_obs_column(::Obs, data, name, column) = add_column_spec(get_obs_spec(data), name, column)
Jobs.add_obs_column(data, name, column) =
	create_spec(DataMatrixFunction(add_obs_column), data, name, column)





counts_fraction_impl_spec(counts, sub_ind, tot_ind; dims) =
	create_spec(SCPCore.counts_fraction, counts, sub_ind, tot_ind; dims, __version=v"0.1.0")

counts_sum_impl_spec(f, counts, ind; dims) =
	create_spec(SCPCore.counts_sum, f, counts, ind; dims, __version=v"0.1.0")


var_counts_fraction(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
var_counts_fraction(::Var, counts, args...; kwargs...) = get_var_spec(counts)
function var_counts_fraction(::Obs, counts, sub_filter, tot_filter, col; project_ids)
	var_spec = get_var_spec(counts)
	sub_ind = prefetched(create_find_matching_ind_spec(sub_filter, var_spec; project_ids))
	tot_ind = prefetched(create_find_matching_ind_spec(tot_filter, var_spec; project_ids))
	values_spec = cached(counts_fraction_impl_spec(get_matrix_spec(counts), sub_ind, tot_ind; dims=1))
	add_column_spec(get_obs_spec(counts), col, values_spec)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_fraction(counts, sub_filter, tot_filter, col; project_ids=:intersect)
	create_spec(DataMatrixFunction(var_counts_fraction), counts, sub_filter, tot_filter, col; project_ids)
end


var_counts_sum(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
var_counts_sum(::Var, counts, args...; kwargs...) = get_var_spec(counts)
function var_counts_sum(::Obs, counts, filter, col; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_spec(filter, get_var_spec(counts); project_ids))
	values_spec = cached(counts_sum_impl_spec(f, get_matrix_spec(counts), ind; dims=1))
	add_column_spec(get_obs_spec(counts), col, values_spec)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_sum(f, counts, filter, col; project_ids=:intersect)
	create_spec(DataMatrixFunction(var_counts_sum), counts, filter, col; f, project_ids)
end
Jobs.var_counts_sum(counts, filter, col; kwargs...) = Jobs.var_counts_sum(identity, counts, filter, col; kwargs...)





# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_fraction(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
function obs_counts_fraction(::Var, counts, sub_filter, tot_filter, col; project_ids)
	obs_spec = get_obs_spec(counts)
	sub_ind = prefetched(create_find_matching_ind_spec(sub_filter, obs_spec; project_ids))
	tot_ind = prefetched(create_find_matching_ind_spec(tot_filter, obs_spec; project_ids))

	values_spec = cached(counts_fraction_impl_spec(get_matrix_spec(counts), sub_ind, tot_ind; dims=2))
	add_column_spec(get_var_spec(counts), col, values_spec)
end
obs_counts_fraction(::Obs, counts, args...; kwargs...) = get_obs_spec(counts)

function Jobs.obs_counts_fraction(counts, sub_filter, tot_filter, col; project_ids=:no)
	create_spec(DataMatrixFunction(obs_counts_fraction), counts, sub_filter, tot_filter, col; project_ids)
end


# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_sum(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
function obs_counts_sum(::Var, counts, filter, col; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_spec(filter, get_obs_spec(counts); project_ids))
	values_spec = cached(counts_sum_impl_spec(f, get_matrix_spec(counts), ind; dims=2))
	add_column_spec(get_var_spec(counts), col, values_spec)
end
obs_counts_sum(::Obs, counts, args...; kwargs...) = get_obs_spec(counts)

function Jobs.obs_counts_sum(f, counts, filter, col; project_ids=:no)
	create_spec(DataMatrixFunction(obs_counts_sum), counts, filter, col; f, project_ids)
end
Jobs.obs_counts_sum(counts, filter, col; kwargs...) = Jobs.obs_counts_sum(identity, counts, filter, col; kwargs...)
