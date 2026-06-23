annotate(::Mat, data; kwargs...) = get_matrix_job(data)
function annotate(f::Union{Var,Obs}, data; kwargs...)
	s = get_job(f, data)
	df = get(kwargs, f isa Var ? :var : :obs, nothing)
	df === nothing && return s
	table_leftjoin_job(s, df)
end

Jobs.annotate_obs(data, df; kwargs...) =
	create_job(DataMatrixFunction(annotate), data; kwargs..., obs=df)
Jobs.annotate_var(data, df; kwargs...) =
	create_job(DataMatrixFunction(annotate), data; kwargs..., var=df)
Jobs.annotate(data, var_df, obs_df; kwargs...) =
	create_job(DataMatrixFunction(annotate), data; kwargs..., var=var_df, obs=obs_df)



add_var_column(f::Union{Mat,Obs}, data, name, column) = get_job(f, data)
add_var_column(::Var, data, name, column) = add_column_job(get_var_job(data), name, column)
Jobs.add_var_column(data, name, column) =
	create_job(DataMatrixFunction(add_var_column), data, name, column)

add_obs_column(f::Union{Mat,Var}, data, name, column) = get_job(f, data)
add_obs_column(::Obs, data, name, column) = add_column_job(get_obs_job(data), name, column)
Jobs.add_obs_column(data, name, column) =
	create_job(DataMatrixFunction(add_obs_column), data, name, column)





counts_fraction_impl_job(counts, sub_ind, tot_ind; dims) =
	create_job(SCPCore.counts_fraction, counts, sub_ind, tot_ind; dims, __version=v"0.1.0")

counts_sum_impl_job(f, counts, ind; dims) =
	create_job(SCPCore.counts_sum, f, counts, ind; dims, __version=v"0.1.0")


var_counts_fraction(::Mat, counts, args...; kwargs...) = get_matrix_job(counts)
var_counts_fraction(::Var, counts, args...; kwargs...) = get_var_job(counts)
function var_counts_fraction(::Obs, counts, col, sub_filter, tot_filter; project_ids)
	var_job = get_var_job(counts)
	sub_ind = prefetched(create_find_matching_ind_job(sub_filter, var_job; project_ids))
	tot_ind = prefetched(create_find_matching_ind_job(tot_filter, var_job; project_ids))
	values_job = cached(counts_fraction_impl_job(get_matrix_job(counts), sub_ind, tot_ind; dims=1))
	add_column_job(get_obs_job(counts), col, values_job)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:intersect)
	create_job(DataMatrixFunction(var_counts_fraction), counts, col, sub_filter, tot_filter; project_ids)
end


var_counts_sum(::Mat, counts, args...; kwargs...) = get_matrix_job(counts)
var_counts_sum(::Var, counts, args...; kwargs...) = get_var_job(counts)
function var_counts_sum(::Obs, counts, col, filter; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_job(filter, get_var_job(counts); project_ids))
	values_job = cached(counts_sum_impl_job(f, get_matrix_job(counts), ind; dims=1))
	add_column_job(get_obs_job(counts), col, values_job)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_sum(f, counts, col::String, filter=Returns(true); project_ids=:intersect)
	create_job(DataMatrixFunction(var_counts_sum), counts, col, filter; f, project_ids)
end
Jobs.var_counts_sum(counts, col::String, args...; kwargs...) = Jobs.var_counts_sum(identity, counts, col, args...; kwargs...)





# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_fraction(::Mat, counts, args...; kwargs...) = get_matrix_job(counts)
function obs_counts_fraction(::Var, counts, col, sub_filter, tot_filter; project_ids)
	obs_job = get_obs_job(counts)
	sub_ind = prefetched(create_find_matching_ind_job(sub_filter, obs_job; project_ids))
	tot_ind = prefetched(create_find_matching_ind_job(tot_filter, obs_job; project_ids))

	values_job = cached(counts_fraction_impl_job(get_matrix_job(counts), sub_ind, tot_ind; dims=2))
	add_column_job(get_var_job(counts), col, values_job)
end
obs_counts_fraction(::Obs, counts, args...; kwargs...) = get_obs_job(counts)

function Jobs.obs_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:no)
	create_job(DataMatrixFunction(obs_counts_fraction), counts, col, sub_filter, tot_filter; project_ids)
end


# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_sum(::Mat, counts, args...; kwargs...) = get_matrix_job(counts)
function obs_counts_sum(::Var, counts, col, filter; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_job(filter, get_obs_job(counts); project_ids))
	values_job = cached(counts_sum_impl_job(f, get_matrix_job(counts), ind; dims=2))
	add_column_job(get_var_job(counts), col, values_job)
end
obs_counts_sum(::Obs, counts, args...; kwargs...) = get_obs_job(counts)

function Jobs.obs_counts_sum(f, counts, col::String, filter=Returns(true); project_ids=:no)
	create_job(DataMatrixFunction(obs_counts_sum), counts, col, filter; f, project_ids)
end
Jobs.obs_counts_sum(counts, col::String, args...; kwargs...) = Jobs.obs_counts_sum(identity, counts, col, args...; kwargs...)
