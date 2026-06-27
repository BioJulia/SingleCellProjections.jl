"""
    SCP.annotate_var(data, df; kwargs...) -> Job

Add variable annotations by left-joining `df` onto `data.var`.
The first column of `df` should contain IDs matching the first column in `data.var`.
The IDs are used as the key when joining the tables.

See also [`annotate_obs`](@ref), [`add_var_column`](@ref).
"""
annotate_var(data, df; kwargs...) =
	create_job(Impl.DataMatrixFunction(Impl.annotate), data; kwargs..., var=df)

"""
    SCP.annotate_obs(data, df; kwargs...) -> Job

Add observation annotations by left-joining `df` onto `data.obs`.
The first column of `df` should contain IDs matching the first column in `data.obs`.
The IDs are used as the key when joining the tables.

See also [`annotate_var`](@ref), [`add_obs_column`](@ref).
"""
annotate_obs(data, df; kwargs...) =
	create_job(Impl.DataMatrixFunction(Impl.annotate), data; kwargs..., obs=df)

"""
    SCP.add_var_column(data, name, column) -> Job

Add a single column named `name` with values `column` to the variable annotations.
The length and order of `column` must match the rows of `data.var`.

See also [`add_obs_column`](@ref), [`annotate_var`](@ref).
"""
add_var_column(data, name, column) =
	create_job(Impl.DataMatrixFunction(Impl.add_var_column), data, name, column)

"""
    SCP.add_obs_column(data, name, column) -> Job

Add a single column named `name` with values `column` to the observation annotations.
The length and order of `column` must match the rows of `data.obs`.

See also [`add_var_column`](@ref), [`annotate_obs`](@ref).
"""
add_obs_column(data, name, column) =
	create_job(Impl.DataMatrixFunction(Impl.add_obs_column), data, name, column)


# TODO: project_ids should it be :yes or :intersect by default???
"""
    SCP.var_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:intersect) -> Job

Compute the fraction of counts from a subset of variables (genes) for each observation,
and add it as a new observation annotation column named `col`.

`sub_filter` and `tot_filter` are predicates applied to the variable annotations to select
the subset and total gene sets respectively.

# Examples

Count the fraction of reads that come from Mitochondrial genes.
```julia
julia> SCP.var_counts_fraction(counts, "fraction_mt", "name"=>startswith("MT-"))
```

See also [`var_counts_sum`](@ref), [`obs_counts_fraction`](@ref).
"""
function var_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:intersect)
	create_job(Impl.DataMatrixFunction(Impl.var_counts_fraction), counts, col, sub_filter, tot_filter; project_ids)
end


# TODO: project_ids should it be :yes or :intersect by default???
"""
    SCP.var_counts_sum([f,] counts, col, filter=Returns(true); project_ids=:intersect) -> Job

Compute the sum of counts (optionally transformed by `f`) from a filtered subset of
variables for each observation, and add it as a new observation annotation column named `col`.

# Examples

Let `counts` be the raw counts.

To count the total number of reads in each cell:
```julia
julia> SCP.var_counts_sum(counts, "total_RNA_count")
```

To count the number of genes that have a non-zero value:
```julia
julia> SCP.var_counts_sum(!iszero, counts, "nonzero_RNA_count")
```

See also [`var_counts_fraction`](@ref), [`obs_counts_sum`](@ref), [`load_counts`](@ref).
"""
function var_counts_sum(f, counts, col::String, filter=Returns(true); project_ids=:intersect)
	create_job(Impl.DataMatrixFunction(Impl.var_counts_sum), counts, col, filter; f, project_ids)
end
var_counts_sum(counts, col::String, args...; kwargs...) = var_counts_sum(identity, counts, col, args...; kwargs...)


"""
    SCP.obs_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:no) -> Job

Compute the fraction of counts from a subset of observations (cells) for each variable,
and add it as a new variable annotation column named `col`.

`sub_filter` and `tot_filter` are predicates applied to the variable annotations to select
the subset and total gene sets respectively.

See also [`obs_counts_sum`](@ref), [`var_counts_fraction`](@ref).
"""
function obs_counts_fraction(counts, col, sub_filter, tot_filter=Returns(true); project_ids=:no)
	create_job(Impl.DataMatrixFunction(Impl.obs_counts_fraction), counts, col, sub_filter, tot_filter; project_ids)
end


"""
    SCP.obs_counts_sum([f,] counts, col, filter=Returns(true); project_ids=:no) -> Job

Compute the sum of counts (optionally transformed by `f`) from a filtered subset of
observations for each variable, and add it as a new variable annotation column named `col`.

# Examples

For each variable, count the number of cells with a non-zero value.
```julia
julia> SCP.obs_counts_sum(!iszero, counts, "nonzero_cell_count")
```

See also [`obs_counts_fraction`](@ref), [`var_counts_sum`](@ref).
"""
function obs_counts_sum(f, counts, col::String, filter=Returns(true); project_ids=:no)
	create_job(Impl.DataMatrixFunction(Impl.obs_counts_sum), counts, col, filter; f, project_ids)
end
obs_counts_sum(counts, col::String, args...; kwargs...) = obs_counts_sum(identity, counts, col, args...; kwargs...)
