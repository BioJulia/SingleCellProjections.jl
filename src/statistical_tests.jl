_create_two_group_prefix(col_name::AbstractString) = string(col_name,'_')
_create_two_group_prefix(col_name, a) = string(col_name,'_',a,'_')
_create_two_group_prefix(col_name, a, b) = string(col_name,'_',a,"_vs_",b,'_')

function _create_ftest_prefix(test, null)
	str = string(join(test,'_'),'_')
	isempty(null) ? str : string(str,"H0_",join(null,'_'),'_')
end

function _create_ttest_prefix(test, null)
	str = string(test,'_')
	isempty(null) ? str : string(str,"H0_",join(null,'_'),'_')
end

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
	statistic_col !== nothing && insertcols!(table, statistic_col=>U; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	table
end

_mannwhitney_table(ref::MatrixRef, args...; kwargs...) =
	_mannwhitney_table(ref.matrix, args...; kwargs...)


"""
	mannwhitney_table(data::DataMatrix, column, [groupA, groupB]; kwargs...)

Perform a Mann-Whitney U-test (also known as a Wilcoxon rank-sum test) between two groups of observations.
The U statistic is corrected for ties, and p-values are computed using a normal approximation.

Note that `data` must be a `DataMatrix` containing a sparse matrix only.
It is recommended to first [`logtransform`](@ref) (or [`tf_idf_transform`](@ref)) the raw counts before performing the Mann-Whitney U-test.

`column` specifies a column in `data.obs` and is used to determine which observations belong in which group.
Observations with `missing` values in the given `column` are always ignored.

If `groupA` and `groupB` are not given, the `column` must contain exactly two unique values (except `missing`).
If `groupA` is given, but not `groupB`, the observations in group A are compared to all other observations (except `missing`).
If both `groupA` and `groupB` are given, the observations in group A are compared the observations in group B.

`mannwhitney_table` returns a Dataframe with columns for variable IDs, U statistics and p-values.

Supported `kwargs` are:
* `statistic_col="U"`   - Name of the output column containing the U statistics. (Set to nothing to remove from output.)
* `pvalue_col="pValue"` - Name of the output column containing the p-values. (Set to nothing to remove from output.)

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
The U statistic is corrected for ties, and p-values are computed using a normal approximation.

Note that `data` must be a `DataMatrix` containing a sparse matrix only.
It is recommended to first [`logtransform`](@ref) (or [`tf_idf_transform`](@ref)) the raw counts before performing the Mann-Whitney U-test.

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
The U statistic is corrected for ties, and p-values are computed using a normal approximation.

Note that `data` must be a `DataMatrix` containing a sparse matrix only.
It is recommended to first [`logtransform`](@ref) (or [`tf_idf_transform`](@ref)) the raw counts before performing the Mann-Whitney U-test.

`mannwhitney` creates a copy of `data` and adds a U statistic and a p-value column to `data.var`.
See [`mannwhitney!`](@ref) and [`mannwhitney_table`](@ref) for more details on groups and `kwargs`.

See also: [`mannwhitney!`](@ref), [`mannwhitney_table`](@ref)
"""
function mannwhitney(data::DataMatrix, args...; var=:copy, obs=:copy, matrix=:keep, kwargs...)
	data = copy(data; var, obs, matrix)
	mannwhitney!(data, args...; kwargs...)
end


# TODO: merge with code in NormalizationModel?
function orthonormal_design(design::DesignMatrix, Q0=nothing; rtol=sqrt(eps()))
	X = design.matrix

	if Q0 !== nothing
		# X  is N×d₁
		# Q0 is N×d₂
		X -= Q0*(Q0'X) # orthogonalize X w.r.t. Q0
	end

	if size(X,2)==1
		# No need to run svd etc. if there just a single column (intercept or t-test column)
		n = norm(X)
		n>rtol && return X./n
		return X[:,1:0] # no columns
	else
		F = svd(X)

		k = something(findlast(>(rtol), F.S), 0)
		return F.U[:,1:k]
	end
end



function _linear_test(data::DataMatrix, test::DesignMatrix, null::DesignMatrix)
	@assert table_cols_equal(data.obs, test.obs_match) "F-test expects design matrix and data matrix observations to be identical."
	@assert table_cols_equal(data.obs, null.obs_match) "F-test expects design matrix and data matrix observations to be identical."

	# TODO: support no null model (not even intercept)
	Q0 = orthonormal_design(null)
	Q1 = orthonormal_design(test, Q0)
	# Q1_pre = orthonormal_design(test, Q0)
	# Q1 = hcat(Q0,Q1_pre) # The purpose of this is to gain numerical accuracy - does it help?

	A = data.matrix

	# fit models
	β0 = A*Q0
	β1 = A*Q1

	# compute residuals
	ssA = variable_sum_squares(A)

	ssβ0 = vec(sum(abs2, β0; dims=2))
	ssβ1 = vec(sum(abs2, β1; dims=2))

	# ssExplained = ssβ1 - ssβ0
	# ssUnexplained = ssA - ssβ1
	# rank0 = size(Q0,2)
	# rank1 = size(Q1,2)

	ssExplained = max.(0.0, ssβ1)
	ssUnexplained = max.(0.0, ssA - ssβ1 - ssβ0)
	rank0 = size(Q0,2)
	rank1 = size(Q1,2)+rank0

	ssExplained, ssUnexplained, rank0, rank1, β1
end


function _ftest_table(data::DataMatrix, test::DesignMatrix, null::DesignMatrix; statistic_col="F", pvalue_col="pValue")
	ssExplained, ssUnexplained, rank0, rank1, _ = _linear_test(data, test, null)
	N = size(data,2)
	ν1 = (rank1-rank0)
	ν2 = (N-rank1)

	if ν1>0 && ν2>0
		F = max.(0.0, (ν2/ν1) * ssExplained./ssUnexplained)
		p = ccdf.(FDist(ν1,ν2), F)
	else
		F = zeros(size(ssExplained))
		p = ones(size(ssExplained))
	end

	table = data.var[:,data.var_id_cols]
	statistic_col !== nothing && insertcols!(table, statistic_col=>F; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	table
end


_splattable(x::Union{Tuple,AbstractVector}) = x
_splattable(x) = (x,)

function ftest_table(data::DataMatrix, test;
                     null=(), center=true, max_categories=nothing, kwargs...)
	test_design = designmatrix(data, _splattable(test)...; center=false, max_categories)
	null_design = designmatrix(data, _splattable(null)...; center, max_categories)

	_ftest_table(data, test_design, null_design; kwargs...)
end

function ftest!(data::DataMatrix, test;
                null=(),
                prefix = _create_ftest_prefix(_splattable(test), _splattable(null)),
                kwargs...)
	df = ftest_table(data, test; null, statistic_col="$(prefix)F", pvalue_col="$(prefix)pValue", kwargs...)
	leftjoin!(data.var, df; on=data.var_id_cols)
	data
end

function ftest(data::DataMatrix, args...; var=:copy, obs=:copy, matrix=:keep, kwargs...)
	data = copy(data; var, obs, matrix)
	ftest!(data, args...; kwargs...)
end





function _ttest_table(data::DataMatrix, test::DesignMatrix, null::DesignMatrix; statistic_col="t", pvalue_col="pValue")
	_, ssUnexplained, rank0, rank1, β1 = _linear_test(data, test, null)
	N = size(data,2)
	ν1 = (rank1-rank0)
	ν2 = (N-rank1)

	if ν1==1 && ν2>0
		t = vec(β1./sqrt.(max.(0.0,(ν1/ν2).*ssUnexplained)))
		p = min.(1.0, 2.0.*ccdf.(TDist(ν2), abs.(t)))
	else
		t = zeros(size(ssUnexplained))
		p = ones(size(ssUnexplained))
	end

	table = data.var[:,data.var_id_cols]
	statistic_col !== nothing && insertcols!(table, statistic_col=>t; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	table
end

function ttest_table(data::DataMatrix, test;
                     null=(), center=true, max_categories=nothing, kwargs...)

	# TODO: support two-group comparison
	test_design = designmatrix(data, test; center=false, max_categories)
	@assert size(test_design.matrix,2)==1

	null_design = designmatrix(data, _splattable(null)...; center, max_categories)

	_ttest_table(data, test_design, null_design; kwargs...)
end


function ttest!(data::DataMatrix, test;
                null=(),
                prefix = _create_ttest_prefix(test, _splattable(null)),
                kwargs...)
	df = ttest_table(data, test; null, statistic_col="$(prefix)t", pvalue_col="$(prefix)pValue", kwargs...)
	leftjoin!(data.var, df; on=data.var_id_cols)
	data
end

function ttest(data::DataMatrix, args...; var=:copy, obs=:copy, matrix=:keep, kwargs...)
	data = copy(data; var, obs, matrix)
	ttest!(data, args...; kwargs...)
end