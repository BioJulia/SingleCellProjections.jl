

# function mannwhitney!(data::DataMatrix; kwargs...)
# end

# function mannwhitney(data::DataMatrix; var=:copy, obs=:copy, matrix=:keep, kwargs...)
# end

# mannwhitney_table(data::DataMatrix{<:MatrixRef}; kwargs...) =

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
                           a::Union{AbstractString,Nothing},
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

function mannwhitney_table(data::DataMatrix, args...; kwargs...)
	groups = _create_two_group(data.obs, args...)
	_mannwhitney_table(data.matrix, data.var[:, data.var_id_cols], groups; kwargs...)
end
