_create_two_group_prefix(col_name::AbstractString) = string(col_name,'_')
_create_two_group_prefix(col_name, a) = string(col_name,'_',a,'_')
_create_two_group_prefix(col_name, a, b) = string(col_name,'_',a,"_vs_",b,'_')

function _create_ftest_prefix(h0, h1)
	str = string(join(h1,'_'),'_')
	isempty(h0) ? str : string(str,"H0_",join(h0,'_'),'_')
end

function _create_ttest_prefix(h0, args...)
	str = _create_two_group_prefix(args...)
	isempty(h0) ? str : string(str,"H0_",join(h0,'_'),'_')
end

function _create_two_group(obs, col_name::AbstractString; h1_missing)
	@assert h1_missing in (:skip,:error)
	col = obs[:,col_name]
	if h1_missing == :error && any(ismissing,col)
		throw(ArgumentError(string("Column \"",col_name,"\" has missing values, set `h1_missing=:skip` to skip them.")))
	end
	unique_values = sort(unique(skipmissing(col))) # Sort to get stability in which group is 1 and which is 2
	if length(unique_values)!=2
		throw(ArgumentError(string("Column \"",col_name,"\" must have exactly two unique values, found ", length(unique_values), ".")))
	end
	groups = zeros(Int, length(col))
	groups[isequal.(col,unique_values[1])] .= 1
	groups[isequal.(col,unique_values[2])] .= 2
	groups
end
function _create_two_group(obs, col_name::AbstractString,
                           a::AbstractString,
                           b::Union{AbstractString,Nothing}=nothing;
                           h1_missing)
	@assert h1_missing in (:skip,:error)
	col = obs[:,col_name]
	if h1_missing == :error && any(ismissing,col)
		throw(ArgumentError(string("Column \"",col_name,"\" has missing values, set `h1_missing=:skip` to skip them.")))
	end
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


function _mannwhitney_table(X::AbstractSparseMatrix, var, groups::Vector{Int};
                            statistic_col="U", pvalue_col="pValue", kwargs...)
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
* `h1_missing` - One of `:skip` and `:error`. If `skip`, missing values in `column` are skipped, otherwise an error is thrown.

The following `kwargs` determine how the computations are threaded:
* `nworkers`      - Number of worker threads used in the computation. Set to 1 to disable threading.
* `chunk_size`    - Number of variables processed in each chunk.
* `channel_size`  - Max number of unprocessed chunks in queue.

See also: [`mannwhitney!`](@ref), [`mannwhitney`](@ref)
"""
function mannwhitney_table(data::DataMatrix, args...; h1_missing=:skip, kwargs...)
	groups = _create_two_group(data.obs, args...; h1_missing)
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


function _add_column_names!(columns, args)
	for c in args
		if c isa CovariateDesc
			c.type != :intercept && push!(columns, c.name)
		else
			push!(columns, c)
		end
	end
end

# No-op of no filtering is needed
function _filter_missing_obs(data::DataMatrix, h1, h0; h1_missing, h0_missing)
	@assert h1_missing in (:skip,:error)
	@assert h0_missing in (:skip,:error)
	h1 = _splattable(h1)
	h0 = _splattable(h0)

	columns = String[]
	h1_missing == :skip && _add_column_names!(columns, h1)
	h0_missing == :skip && _add_column_names!(columns, h0)

	isempty(columns) && return data

	mask = completecases(data.obs, unique(columns))

	all(mask) && return data
	return data[:,mask]
end


# TODO: merge with code in NormalizationModel?
# returns the orthogonalized design matrix, and if it was a single column design matrix - return the norm before rescaling (otherwise return 0.0)
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
		n>rtol && return X./n, n
		return X[:,1:0], 0.0 # no columns
	else
		F = svd(X)

		k = something(findlast(>(rtol), F.S), 0)
		return F.U[:,1:k], 0.0
	end
end



function _linear_test(data::DataMatrix, h1::DesignMatrix, h0::DesignMatrix)
	@assert table_cols_equal(data.obs, h1.obs_match) "Design matrix and data matrix observations should be identical."
	@assert table_cols_equal(data.obs, h0.obs_match) "Design matrix (H0) and data matrix observations should be identical."

	# TODO: support no null model (not even intercept)
	Q0,_ = orthonormal_design(h0)
	Q1,scale = orthonormal_design(h1, Q0)
	# Q1_pre = orthonormal_design(h1, Q0)
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

	ssExplained, ssUnexplained, rank0, rank1, β1, scale
end


function _ftest_table(data::DataMatrix, h1::DesignMatrix, h0::DesignMatrix;
                      statistic_col="F", pvalue_col="pValue")
	ssExplained, ssUnexplained, rank0, rank1, _, _ = _linear_test(data, h1, h0)
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

function ftest_table(data::DataMatrix, h1;
                     h0=(),
                     h1_missing=:skip, h0_missing=:error,
                     center=true, max_categories=nothing, kwargs...)
	h1 = _splattable(h1)
	h0 = _splattable(h0)
	data = _filter_missing_obs(data, h1, h0; h1_missing, h0_missing)

	h1_design = designmatrix(data, h1...; center=false, max_categories)
	h0_design = designmatrix(data, h0...; center, max_categories)

	_ftest_table(data, h1_design, h0_design; kwargs...)
end

function ftest!(data::DataMatrix, h1;
                h0=(),
                prefix = _create_ftest_prefix(_splattable(h0), _splattable(h1)),
                kwargs...)
	df = ftest_table(data, h1; h0, statistic_col="$(prefix)F", pvalue_col="$(prefix)pValue", kwargs...)
	leftjoin!(data.var, df; on=data.var_id_cols)
	data
end

function ftest(data::DataMatrix, args...; var=:copy, obs=:copy, matrix=:keep, kwargs...)
	data = copy(data; var, obs, matrix)
	ftest!(data, args...; kwargs...)
end





function _ttest_table(data::DataMatrix, h1::DesignMatrix, h0::DesignMatrix;
                      statistic_col="t", pvalue_col="pValue", difference_col="difference")
	_, ssUnexplained, rank0, rank1, β1, scale = _linear_test(data, h1, h0)
	N = size(data,2)
	ν1 = (rank1-rank0)
	ν2 = (N-rank1)

	if ν1==1 && ν2>0
		t = vec(β1./sqrt.(max.(0.0,(ν1/ν2).*ssUnexplained)))
		p = min.(1.0, 2.0.*ccdf.(TDist(ν2), abs.(t)))
		d = vec(β1)/(scale*_covariate_scale(only(h1.covariates)))
	else
		t = zeros(size(ssUnexplained))
		p = ones(size(ssUnexplained))
		d = zeros(size(ssUnexplained))
	end

	table = data.var[:,data.var_id_cols]
	statistic_col !== nothing && insertcols!(table, statistic_col=>t; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	difference_col !== nothing && insertcols!(table, difference_col=>d; copycols=false)
	table
end

function ttest_table(data::DataMatrix, h1::CovariateDesc;
                     h0=(),
                     h1_missing=:skip, h0_missing=:error,
                     center=true, max_categories=nothing, kwargs...)
	h0 = _splattable(h0)
	data = _filter_missing_obs(data, h1, h0; h1_missing, h0_missing)

	h1_design = designmatrix(data, h1; center=false, max_categories)
	@assert size(h1_design.matrix,2)==1

	h0_design = designmatrix(data, h0...; center, max_categories)

	_ttest_table(data, h1_design, h0_design; kwargs...)
end

# Handle Two-Group
function ttest_table(data::DataMatrix, h1; kwargs...)
	t = eltype(data.obs[!,h1]) <: Union{Missing,Number} ? :numerical : :twogroup
	ttest_table(data, covariate(h1, t); kwargs...)
end
ttest_table(data::DataMatrix, h1, groupA, groupB=nothing; kwargs...) =
	ttest_table(data, covariate(h1, groupA, groupB); kwargs...)


function ttest!(data::DataMatrix, args...;
                h0=(),
                prefix = _create_ttest_prefix(_splattable(h0), args...),
                kwargs...)
	df = ttest_table(data, args...; h0, statistic_col="$(prefix)t", pvalue_col="$(prefix)pValue", difference_col="$(prefix)difference", kwargs...)
	leftjoin!(data.var, df; on=data.var_id_cols)
	data
end

function ttest(data::DataMatrix, args...; var=:copy, obs=:copy, matrix=:keep, kwargs...)
	data = copy(data; var, obs, matrix)
	ttest!(data, args...; kwargs...)
end