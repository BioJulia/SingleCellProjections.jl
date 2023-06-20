_create_two_group_prefix(col_name::AbstractString) = string(col_name,'_')
_create_two_group_prefix(col_name, a) = string(col_name,'_',a,'_')
_create_two_group_prefix(col_name, a, b) = string(col_name,'_',a,"_vs_",b,'_')

function _create_two_group(obs, col_name::AbstractString)
	col = obs[:,col_name]
	unique_values = sort(unique(skipmissing(col))) # Sort to get stability in which group is 1 and which is 2
	if length(unique_values)!=2
		throw(ArgumentError(string("Column \"",col_name,"\" must have exactly two unique values (ignoring missing), found ", length(unique_values), ".")))
	end
	groups = zeros(Int, length(col))
	groups[isequal.(col,unique_values[1])] .= 1
	groups[isequal.(col,unique_values[2])] .= 2
	groups
end
function _create_two_group(obs, col_name::AbstractString,
                           a::AbstractString,
                           b::Union{AbstractString,Nothing}=nothing)
	col = obs[:,col_name]
	groups = zeros(Int, length(col))
	a in col || throw(ArgumentError(string("Column \"",col_name,"\" doesn't contain \"",a,"\".")))
	groups[isequal.(col,a)] .= 1
	if b !== nothing
		b in col || throw(ArgumentError(string("Column \"",col_name,"\" doesn't contain \"",b,"\".")))
		groups[isequal.(col,b)] .= 2
	else
		groups[.!isequal.(col,a) .& .!ismissing.(col)] .= 2
	end
	groups
end


function _mannwhitney_table(X::AbstractSparseMatrix, var, groups::Vector{Int}; statistic_col="U", pvalue_col="pValue", kwargs...)
	U,p = mannwhitney_sparse(X::AbstractSparseMatrix, groups; kwargs...)
	table = copy(var)
	insertcols!(table, statistic_col=>U, pvalue_col=>p; copycols=false)
end

_mannwhitney_table(ref::MatrixRef, args...; kwargs...) =
	_mannwhitney_table(ref.matrix, args...; kwargs...)


"""
	mannwhitney_table(data::DataMatrix, column, [groupA, groupB]; kwargs...)

Perform a Mann-Whitney U-test (also known as a Wilcoxon rank-sum test) between two groups of observations.

`column` specifies a column in `data.obs` and is used to determine which observations belong in which group.
Observations with `missing` values in the given `column` are always ignored.

If `groupA` and `groupB` are not given, the `column` must contain exactly two unique values (except `missing`).
If `groupA` is given, but not `groupB`, the observations in group A are compared to all other observations (except `missing`).
If both `groupA` and `groupB` are given, the observations in group A are compared the observations in group B.

`mannwhitney_table` returns a Dataframe with columns for variable IDs, U statistics and p-values.

Supported `kwargs` are:
* `statistic_col="U"`   - Name of the output column containing the U statistics.
* `pvalue_col="pValue"` - Name of the output column containing the p-values.

The following `kwargs` determine how the computations are threaded:
* `nworkers`      - Number of worker threads used in the computation. Set to 1 to disable threading.
* `chunk_size`    - Number of variables processed in each chunk.
* `channel_size`  - Max number of unprocessed chunks in queue.

See also: [`mannwhitney!`](@ref), [`mannwhitney`](@ref)
"""
function mannwhitney_table(data::DataMatrix, args...; kwargs...)
	groups = _create_two_group(data.obs, args...)
	_mannwhitney_table(data.matrix, data.var[:, data.var_id_cols], groups; kwargs...)
end


"""
	mannwhitney!(data::DataMatrix, column, [groupA, groupB]; kwargs...)

Perform a Mann-Whitney U-test (also known as a Wilcoxon rank-sum test) between two groups of observations.

`mannwhitney!` adds a U statistic and a p-value column to `data.var`.
See [`mannwhitney_table`](@ref) for more details on groups and kwargs.

In addition `mannwhitney!` supports the `kwarg`:
* `prefix` - Output column names for U statistics and p-values will be prefixed with this string. If none is given, it will be constructed from `column`, `groupA` and `groupB`.

See also: [`mannwhitney_table`](@ref), [`mannwhitney`](@ref)
"""
function mannwhitney!(data::DataMatrix, args...;
                      prefix = _create_two_group_prefix(args...),
                      kwargs...)
	df = mannwhitney_table(data, args...; statistic_col="$(prefix)U", pvalue_col="$(prefix)pValue", kwargs...)
	leftjoin!(data.var, df; on=data.var_id_cols)
	data
end


"""
	mannwhitney(data::DataMatrix, column, [groupA, groupB]; var=:copy, obs=:copy, matrix=:keep, kwargs...)

Perform a Mann-Whitney U-test (also known as a Wilcoxon rank-sum test) between two groups of observations.

`mannwhitney` creates a copy of `data` and adds a U statistic and a p-value column to `data.var`.
See [`mannwhitney!`](@ref) and [`mannwhitney_table`](@ref) for more details on groups and `kwargs`.

See also: [`mannwhitney!`](@ref), [`mannwhitney_table`](@ref)
"""
function mannwhitney(data::DataMatrix, args...; var=:copy, obs=:copy, matrix=:keep, kwargs...)
	data = copy(data; var, obs, matrix)
	mannwhitney!(data, args...; kwargs...)
end
