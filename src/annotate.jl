annotate(::Mat, data; kwargs...) = get_matrix_job(data)
function annotate(f::Union{Var,Obs}, data; kwargs...)
	s = get_job(f, data)
	df = get(kwargs, f isa Var ? :var : :obs, nothing)
	df === nothing && return s
	table_leftjoin_job(s, df)
end

"""
    Jobs.annotate_var(data, df; kwargs...) -> Job

Add variable annotations by left-joining `df` onto `data.var`.
The first column of `df` should contain IDs matching the first column in `data.var`.
The IDs are used as the key when joining the tables.

See also `Jobs.annotate_obs`, `Jobs.annotate`, `Jobs.add_var_column`.
"""
Jobs.annotate_var(data, df; kwargs...) =
	create_job(DataMatrixFunction(annotate), data; kwargs..., var=df)

"""
    Jobs.annotate_obs(data, df; kwargs...) -> Job

Add observation annotations by left-joining `df` onto `data.obs`.
The first column of `df` should contain IDs matching the first column in `data.obs`.
The IDs are used as the key when joining the tables.

See also `Jobs.annotate_var`, `Jobs.annotate`, `Jobs.add_obs_column`.
"""
Jobs.annotate_obs(data, df; kwargs...) =
	create_job(DataMatrixFunction(annotate), data; kwargs..., obs=df)



add_var_column(f::Union{Mat,Obs}, data, name, column) = get_job(f, data)
add_var_column(::Var, data, name, column) = add_column_job(get_var_job(data), name, column)
"""
    Jobs.add_var_column(data, name, column) -> Job

Add a single column named `name` with values `column` to the variable annotations.
The length and order of `column` must match the rows of `data.var`.

See also `Jobs.add_obs_column`, `Jobs.annotate_var`.
"""
Jobs.add_var_column(data, name, column) =
	create_job(DataMatrixFunction(add_var_column), data, name, column)

add_obs_column(f::Union{Mat,Var}, data, name, column) = get_job(f, data)
add_obs_column(::Obs, data, name, column) = add_column_job(get_obs_job(data), name, column)
"""
    Jobs.add_obs_column(data, name, column) -> Job

Add a single column named `name` with values `column` to the observation annotations.
The length and order of `column` must match the rows of `data.obs`.

See also `Jobs.add_var_column`, `Jobs.annotate_obs`.
"""
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
"""
    Jobs.var_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:intersect) -> Job

Compute the fraction of counts from a subset of variables (genes) for each observation,
and add it as a new observation annotation column named `col`.

`sub_filter` and `tot_filter` are predicates applied to the variable annotations to select
the subset and total gene sets respectively.

# Examples

Count the fraction of reads that come from Mitochondrial genes.
```julia
julia> var_counts_fraction(counts, "fraction_mt", "name"=>startswith("MT-"))
```

See also `Jobs.var_counts_sum`, `Jobs.obs_counts_fraction`.
"""
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
"""
    Jobs.var_counts_sum([f,] counts, col, filter=Returns(true); project_ids=:intersect) -> Job

Compute the sum of counts (optionally transformed by `f`) from a filtered subset of
variables for each observation, and add it as a new observation annotation column named `col`.

# Examples

Let `counts` be the raw counts.

To count the total number of reads in each cell:
```julia
julia> Jobs.var_counts_sum(counts, "total_RNA_count")
```

To count the number of genes that have a non-zero value:
```julia
julia> Jobs.var_counts_sum(!iszero, counts, "nonzero_RNA_count")
```

See also `Jobs.var_counts_fraction`, `Jobs.obs_counts_sum`, [`Jobs.load_counts`](@ref).
"""
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

"""
    Jobs.obs_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:no) -> Job

Compute the fraction of counts from a subset of observations (cells) for each variable,
and add it as a new variable annotation column named `col`.

`sub_filter` and `tot_filter` are predicates applied to the variable annotations to select
the subset and total gene sets respectively.

See also [`Jobs.obs_counts_sum`](@ref), [`Jobs.var_counts_fraction`](@ref).
"""
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

"""
    Jobs.obs_counts_sum([f,] counts, col, filter=Returns(true); project_ids=:no) -> Job

Compute the sum of counts (optionally transformed by `f`) from a filtered subset of
observations for each variable, and add it as a new variable annotation column named `col`.

# Examples

For each variable, count the number of cells with a non-zero value.
```julia
julia> Jobs.obs_counts_sum(!iszero, counts, "nonzero_cell_count")
```

See also `Jobs.obs_counts_fraction`, `Jobs.var_counts_sum`.
"""
function Jobs.obs_counts_sum(f, counts, col::String, filter=Returns(true); project_ids=:no)
	create_job(DataMatrixFunction(obs_counts_sum), counts, col, filter; f, project_ids)
end
Jobs.obs_counts_sum(counts, col::String, args...; kwargs...) = Jobs.obs_counts_sum(identity, counts, col, args...; kwargs...)
