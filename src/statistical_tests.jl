_splattable(x::Union{Tuple,AbstractVector}) = x
_splattable(x) = (x,)

function _create_ftest_prefix(h0, h1)
	str = join(covariate_prefix.(h1))
	isempty(h0) ? str : string(str,"H0_",join(covariate_prefix.(h0)))
end

function _create_ttest_prefix(h0, h1)
	str = covariate_prefix(h1)
	isempty(h0) ? str : string(str,"H0_",join(covariate_prefix.(h0)))
end

_create_mannwhitney_prefix(h1) = covariate_prefix(h1)

function _create_mannwhitney_groups(obs, h1::CovariateDesc{T}; h1_missing) where T
	@assert h1.type == :twogroup
	@assert h1_missing in (:skip,:error)

	name, group_a, group_b = h1.name, h1.group_a, h1.group_b
	v = obs[!,name]

	if h1_missing == :error && any(ismissing,v)
		throw(ArgumentError(string("Column \"",name,"\" has missing values, set `h1_missing=:skip` to skip them.")))
	end

	# Three cases
	# group_a and group_b given - must have at least one group_a and at least one group_b (missing excluded)
	# group_a given             - must have at least one group_a and at least one !group_a (missing excluded)
	# none given                - must have exactly two groups (missing excluded)

	uv = unique(skipmissing(v))

	if group_a === nothing
		@assert group_b === nothing
		if length(uv)!=2
			throw(ArgumentError(string("Column \"",name,"\" must have exactly two unique values, found; ", uv, ".")))
		end
		group_a, group_b = minmax(uv[1], uv[2])
	end

	maskA = isequal.(v, group_a)
	if !any(maskA)
		throw(ArgumentError(string("Column \"",name,"\" doesn't contain \"",group_a,"\".")))
	end

	if group_b !== nothing
		maskB = isequal.(v, group_b)
		if !any(maskB)
			throw(ArgumentError(string("Column \"",name,"\" only contains one group: \"",group_a,"\".")))
		end
	else
		maskB = .!isequal.(v, group_a) .& .!ismissing.(v)
		if !any(maskB)
			throw(ArgumentError(string("Column \"",name,"\" doesn't contain \"",group_b,"\".")))
		end
	end

	groups = zeros(Int, size(obs,1))
	groups[maskA] .= 1
	groups[maskB] .= 2
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


_mannwhitney_covariate(h1::CovariateDesc) = h1
_mannwhitney_covariate(h1) = covariate(h1, :twogroup)
_mannwhitney_covariate(h1, args...) = covariate(h1, args...)


"""
	mannwhitney_table(data::DataMatrix, column, [group_a, group_b]; kwargs...)

Perform a Mann-Whitney U-test (also known as a Wilcoxon rank-sum test) between two groups of observations.
The U statistic is corrected for ties, and p-values are computed using a normal approximation.

Note that `data` must be a `DataMatrix` containing a sparse matrix only.
It is recommended to first [`logtransform`](@ref) (or [`tf_idf_transform`](@ref)) the raw counts before performing the Mann-Whitney U-test.

`column` specifies a column in `data.obs` and is used to determine which observations belong in which group.

If `group_a` and `group_b` are not given, the `column` must contain exactly two unique values (except `missing`).
If `group_a` is given, but not `group_b`, the observations in group A are compared to all other observations (except `missing`).
If both `group_a` and `group_b` are given, the observations in group A are compared the observations in group B.

`mannwhitney_table` returns a Dataframe with columns for variable IDs, U statistics and p-values.

Supported `kwargs` are:
* `statistic_col="U"`   - Name of the output column containing the U statistics. (Set to nothing to remove from output.)
* `pvalue_col="pValue"` - Name of the output column containing the p-values. (Set to nothing to remove from output.)
* `h1_missing=:skip`    - One of `:skip` and `:error`. If `skip`, missing values in `column` are skipped, otherwise an error is thrown.

The following `kwargs` determine how the computations are threaded:
* `nworkers`      - Number of worker threads used in the computation. Set to 1 to disable threading.
* `chunk_size`    - Number of variables processed in each chunk.
* `channel_size`  - Max number of unprocessed chunks in queue.

See also: [`mannwhitney!`](@ref), [`mannwhitney`](@ref)
"""
function mannwhitney_table(data::DataMatrix, args...; h1_missing=:skip, kwargs...)
	h1 = _mannwhitney_covariate(args...)
	groups = _create_mannwhitney_groups(data.obs, h1; h1_missing)
	_mannwhitney_table(data.matrix, select(data.var,1), groups; kwargs...)
end


"""
	mannwhitney!(data::DataMatrix, column, [group_a, group_b]; kwargs...)

Perform a Mann-Whitney U-test (also known as a Wilcoxon rank-sum test) between two groups of observations.
The U statistic is corrected for ties, and p-values are computed using a normal approximation.

Note that `data` must be a `DataMatrix` containing a sparse matrix only.
It is recommended to first [`logtransform`](@ref) (or [`tf_idf_transform`](@ref)) the raw counts before performing the Mann-Whitney U-test.

`mannwhitney!` adds a U statistic and a p-value column to `data.var`.
See [`mannwhitney_table`](@ref) for more details on groups and kwargs.

In addition `mannwhitney!` supports the `kwarg`:
* `prefix` - Output column names for U statistics and p-values will be prefixed with this string. If none is given, it will be constructed from `column`, `group_a` and `group_b`.

See also: [`mannwhitney_table`](@ref), [`mannwhitney`](@ref)
"""
function mannwhitney!(data::DataMatrix, args...;
                      prefix = nothing,
                      kwargs...)
	h1 = _mannwhitney_covariate(args...)
	prefix === nothing && (prefix = _create_mannwhitney_prefix(h1))
	df = mannwhitney_table(data, h1; statistic_col="$(prefix)U", pvalue_col="$(prefix)pValue", kwargs...)
	leftjoin!(data.var, df; on=names(data.var,1))
	data
end


"""
	mannwhitney(data::DataMatrix, column, [group_a, group_b]; var=:copy, obs=:copy, matrix=:keep, kwargs...)

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

function _keep_mask(df, c::CovariateDesc{T}) where T
	if c.type == :intercept
		return trues(size(df,1))
	elseif c.type == :twogroup && c.group_b !== nothing
		return isequal.(df[!,c.name], c.group_a) .| isequal.(df[!,c.name], c.group_b)
	else
		return completecases(df,c.name)
	end
end

# No-op of no filtering is needed
function _filter_missing_obs(data::DataMatrix, h1, h0; h1_missing, h0_missing)
	@assert h1_missing in (:skip,:error)
	@assert h0_missing in (:skip,:error)

	mask = trues(size(data,2))
	h1_missing == :skip && (mask=mapreduce(c->_keep_mask(data.obs, c), .&, h1; init = mask))
	h0_missing == :skip && (mask=mapreduce(c->_keep_mask(data.obs, c), .&, h0; init = mask))

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

	# TODO: Support no null model? (not even intercept)
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

	table = select(data.var,1)
	statistic_col !== nothing && insertcols!(table, statistic_col=>F; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	table
end


"""
	ftest_table(data::DataMatrix, h1; h0, kwargs...)

Performs an F-Test with the given `h1` (alternative hypothesis) and `h0` (null hypothesis).
Examples of F-Tests are ANOVA and Quadratic Regression, but any linear model can be used.
(See "Examples" below for concrete examples.)

F-tests can be performed on any `DataMatrix`, but it is almost always recommended to do it directly after transforming the data using e.g. `sctransform`, `logtransform` or `tf_idf_transform`.

!!! note "Normalization"
    Do not use `ftest_table` after normalizing the data using `normalize_matrix`: `ftest_table` needs to know about the `h0` model (regressed out covariates) for correction computations. Failing to do so can result in incorrect results.
    If you want to correct for the same covariates, pass them as `h0` to `ftest_table`.

`h1` can be:
* A string specifying a column name of `data.obs`. Auto-detection determines if the column is categorical (ANOVA) or numerical.
* A [`covariate`](@ref) for more control of how to interpret the values in a column.
* A tuple or vector of the above for compound models.

`ftest_table` returns a Dataframe with columns for variable IDs, F-statistics and p-values.

Supported `kwargs` are:
* `h0`                  - Use a non-trivial `h0` (null) model. Specified in the same way as `h1` above.
* `center=true`         - Add an intercept to the `h0` (null) model.
* `statistic_col="F"`   - Name of the output column containing the F-statistics. (Set to nothing to remove from output.)
* `pvalue_col="pValue"` - Name of the output column containing the p-values. (Set to nothing to remove from output.)
* `h1_missing=:skip`    - One of `:skip` and `:error`. If `skip`, missing values in `h1` columns are skipped, otherwise an error is thrown.
* `h0_missing=:error`   - One of `:skip` and `:error`. If `skip`, missing values in `h0` columns are skipped, otherwise an error is thrown.
* `allow_normalized_matrix=false` - Set to true to accept running on a `DataMatrix` that has been normalized.

# Examples

Perform an ANOVA using the "celltype" annotation.
```julia
julia> ftest_table(transformed, "celltype")
```

Perform an ANOVA using the "celltype" annotation, while correcting for `fraction_mt` (a linear covariate).
```julia
julia> ftest_table(transformed, "celltype"; h0="fraction_mt")
```

Perform an ANOVA using the "celltype" annotation, while correcting for `fraction_mt` (a linear covariate) and "phase" (a categorical covariate).
```julia
julia> ftest_table(transformed, "celltype"; h0=("fraction_mt","phase"))
```

Perform Quadractic Regression using the covariate `x`, by first creating an annotation for `x` squared, and then using a compound model.
```julia
julia> data.obs.x2 = data.obs.x.^2;

julia> ftest_table(transformed, ("x","x2"))
```

See also: [`ftest!`](@ref), [`ftest`](@ref), [`ttest_table`](@ref), [`covariate`](@ref)
"""
function ftest_table(data::DataMatrix, h1;
                     h0=(),
                     h1_missing=:skip, h0_missing=:error,
                     center=true, max_categories=nothing,
                     allow_normalized_matrix=false,
                     kwargs...)
	if !allow_normalized_matrix && any(x->x isa NormalizationModel, data.models)
		throw(ArgumentError("It is not recommended to perform an F-test after normalizing the data matrix. Set allow_normalized_matrix=true to disable this error."))
	end

	h1 = covariate.(_splattable(h1))
	h0 = covariate.(_splattable(h0))
	data = _filter_missing_obs(data, h1, h0; h1_missing, h0_missing)

	h1_design = designmatrix(data, h1...; center=false, max_categories)
	h0_design = designmatrix(data, h0...; center, max_categories)

	_ftest_table(data, h1_design, h0_design; kwargs...)
end

"""
	ftest!(data::DataMatrix, h1; h0, kwargs...)

Performs an F-Test with the given `h1` (alternative hypothesis) and `h0` (null hypothesis).
Examples of F-Tests are ANOVA and Quadratic Regression, but any linear model can be used.

`ftest!` adds a F-statistic and a p-value column to `data.var`.

See [`ftest_table`](@ref) for usage examples and more details on computations and parameters.

In addition `ftest!` supports the `kwarg`:
* `prefix` - Output column names for F-statistics and p-values will be prefixed with this string. If none is given, it will be constructed from `h1` and `h0`.

See also: [`ftest_table`](@ref), [`ftest`](@ref), [`ttest!`](@ref)
"""
function ftest!(data::DataMatrix, h1;
                h0=(),
                prefix = nothing,
                kwargs...)
	h1 = covariate.(_splattable(h1))
	h0 = covariate.(_splattable(h0))
	prefix === nothing && (prefix = _create_ftest_prefix(h0, h1))
	df = ftest_table(data, h1; h0, statistic_col="$(prefix)F", pvalue_col="$(prefix)pValue", kwargs...)
	leftjoin!(data.var, df; on=names(data.var,1))
	data
end

"""
	ftest(data::DataMatrix, h1; h0, var=:copy, obs=:copy, matrix=:keep, kwargs...)

Performs an F-Test with the given `h1` (alternative hypothesis) and `h0` (null hypothesis).
Examples of F-Tests are ANOVA and Quadratic Regression, but any linear model can be used.

`ftest` creates a copy of `data` and adds a F-statistic and a p-value column to `data.var`.

See [`ftest_table`](@ref) and [`ftest!`](@ref) for usage examples and more details on computations and parameters.

See also: [`ftest!`](@ref), [`ftest_table`](@ref), [`ttest`](@ref)
"""
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

	table = select(data.var,1)
	statistic_col !== nothing && insertcols!(table, statistic_col=>t; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	difference_col !== nothing && insertcols!(table, difference_col=>d; copycols=false)
	table
end


# Handle Two-Group
_ttest_covariate(::Any, h1::CovariateDesc) = h1
function _ttest_covariate(data, h1)
	t = eltype(data.obs[!,h1]) <: Union{Missing,Number} ? :numerical : :twogroup
	covariate(h1, t)
end
_ttest_covariate(::Any, h1, group_a, group_b=nothing) = covariate(h1, group_a, group_b)


"""
	ttest_table(data::DataMatrix, h1, [group_a], [group_b]; h0, kwargs...)

Performs a t-Test with the given `h1` (alternative hypothesis) and `h0` (null hypothesis).
Examples of t-Tests are Two-Group tests and Linear Regression.

T-tests can be performed on any `DataMatrix`, but it is almost always recommended to do it directly after transforming the data using e.g. `sctransform`, `logtransform` or `tf_idf_transform`.

!!! note "Normalization"
    Do not use `ttest_table` after normalizing the data using `normalize_matrix`: `ttest_table` needs to know about the `h0` model (regressed out covariates) for correction computations. Failing to do so can result in incorrect results.
    If you want to correct for the same covariates, pass them as `h0` to `ttest_table`.

`h1` can be:
* A string specifying a column name of `data.obs`. Auto-detection determines if the column is categorical (Two-Group) or numerical (linear regression).
  - If `group_a` and `group_b` are specified, a Two-Group test between `group_a` and `group_b` is performed.
  - If `group_a` is specified, but not `group_b`, a Two-Group test between `group_a` and all other observations is performed.
* A [`covariate`](@ref) for more control of how to interpret the values in the column.

`ttest_table` returns a Dataframe with columns for variable IDs, t-statistics, p-values and differences.
For Two-group tests, `difference` is the difference in mean between the two groups.
For linear regression, the difference corresponds to the rate of change.

Supported `kwargs` are:
* `h0`                            - Use a non-trivial `h0` (null) model. Specified in the same way as `h1` above.
* `center=true`                   - Add an intercept to the `h0` (null) model.
* `statistic_col="t"`             - Name of the output column containing the t-statistics. (Set to nothing to remove from output.)
* `pvalue_col="pValue"`           - Name of the output column containing the p-values. (Set to nothing to remove from output.)
* `difference_col="difference"`   - Name of the output column containing the differences. (Set to nothing to remove from output.)
* `h1_missing=:skip`              - One of `:skip` and `:error`. If `skip`, missing values in `h1` columns are skipped, otherwise an error is thrown.
* `h0_missing=:error`             - One of `:skip` and `:error`. If `skip`, missing values in `h0` columns are skipped, otherwise an error is thrown.
* `allow_normalized_matrix=false` - Set to true to accept running on a `DataMatrix` that has been normalized.

# Examples

Perform a Two-Group t-test between celltypes "Mono" and "DC".
```julia
julia> ttest_table(transformed, "celltype", "Mono", "DC")
```

Perform a Two-Group t-test between celltype "Mono" and all other cells.
```julia
julia> ttest_table(transformed, "celltype", "Mono")
```

Perform a Two-Group t-test between celltypes "Mono" and "DC", while correcting for "fraction_mt" (a linear covariate).
```julia
julia> ttest_table(transformed, "celltype", "Mono", "DC")
```

Perform Linear Regression using the covariate "fraction_mt".
```julia
julia> ttest_table(transformed, "fraction_mt")
```

See also: [`ttest!`](@ref), [`ttest`](@ref), [`ftest_table`](@ref), [`mannwhitney_table`](@ref), [`covariate`](@ref)
"""
function ttest_table(data::DataMatrix, args...;
                     h0=(),
                     h1_missing=:skip, h0_missing=:error,
                     center=true, max_categories=nothing,
                     allow_normalized_matrix=false,
                     kwargs...)
	if !allow_normalized_matrix && any(x->x isa NormalizationModel, data.models)
		throw(ArgumentError("It is not recommended to perform a t-test after normalizing the data matrix. Set allow_normalized_matrix=true to disable this error."))
	end

	h1 = _ttest_covariate(data, args...)
	h0 = covariate.(_splattable(h0))

	data = _filter_missing_obs(data, (h1,), h0; h1_missing, h0_missing)

	h1_design = designmatrix(data, h1; center=false, max_categories)
	@assert size(h1_design.matrix,2)==1

	h0_design = designmatrix(data, h0...; center, max_categories)

	_ttest_table(data, h1_design, h0_design; kwargs...)
end


"""
	ttest!(data::DataMatrix, h1, [group_a], [group_b]; h0, kwargs...)

Performs a t-Test with the given `h1` (alternative hypothesis) and `h0` (null hypothesis).
Examples of t-Tests are Two-Group tests and Linear Regression.

`ttest!` adds a t-statistic, a p-value and a difference column to `data.var`.

See [`ttest_table`](@ref) for usage examples and more details on computations and parameters.

In addition `ttest!` supports the `kwarg`:
* `prefix` - Output column names for t-statistics, p-values and differences will be prefixed with this string. If none is given, it will be constructed from `h1`, `group_a`, `group_b` and `h0`.

See also: [`ttest_table`](@ref), [`ttest`](@ref), [`ftest!`](@ref), [`mannwhitney!`](@ref)
"""
function ttest!(data::DataMatrix, args...;
                h0=(),
                prefix = nothing,
                kwargs...)
	h1 = _ttest_covariate(data, args...)
	h0 = covariate.(_splattable(h0))
	prefix === nothing && (prefix = _create_ttest_prefix(h0, h1))

	df = ttest_table(data, h1; h0, statistic_col="$(prefix)t", pvalue_col="$(prefix)pValue", difference_col="$(prefix)difference", kwargs...)
	leftjoin!(data.var, df; on=names(data.var,1))
	data
end

"""
	ttest(data::DataMatrix, h1, [group_a], [group_b]; h0, var=:copy, obs=:copy, matrix=:keep, kwargs...)

Performs a t-Test with the given `h1` (alternative hypothesis) and `h0` (null hypothesis).
Examples of t-Tests are Two-Group tests and Linear Regression.

`ttest` creates a copy of `data` and adds a t-statistic, a p-value and a difference column to `data.var`.

See [`ttest_table`](@ref) and [`ttest!`](@ref) for usage examples and more details on computations and parameters.

See also: [`ttest!`](@ref), [`ttest_table`](@ref), [`ftest`](@ref), [`mannwhitney`](@ref)
"""
function ttest(data::DataMatrix, args...; var=:copy, obs=:copy, matrix=:keep, kwargs...)
	data = copy(data; var, obs, matrix)
	ttest!(data, args...; kwargs...)
end