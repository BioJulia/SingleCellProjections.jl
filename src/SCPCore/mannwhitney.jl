"""
	ustatistic_single(X, j, groups, n1, n2)

NB: Assumes all sparse non-zeros are positive.

`X` is a sparse matrix where each column is a variable.
`j` is the current variable.
`groups` is a vector with values: `1` for each sample in group 1, `2` for each sample in group 2 and `0` for samples in neither group.
`n1` number of elements in group 1 (precomputed from `groups`)
`n2` number of elements in group 2 (precomputed from `groups`)
"""
function ustatistic_single(X::AbstractSparseMatrix{T}, j, groups, n1, n2) where T
	@assert size(X,1)==length(groups)

	V = nonzeros(X)
	R = rowvals(X)

	# TODO: reuse scratch space between calls to avoid excessive allocations
	values = Tuple{T,Bool}[] # value, inGroupOne

	# gather values that are in group 1 and 2
	for k in nzrange(X,j)
		i = R[k]
		v = V[k]
		g = groups[i]
		if g==1 || g==2
			push!(values, (v,g==1))
		end
	end

	Rtimes2 = 0 # Due to ties, possible values are of the form k/2. We thus store R*2 here, to be able to work with integers.
	tie_adjustment = 0.0 # accumulate t³-t where t is the number of ties for each unique rank
	nz_count1 = 0 # total number of non-zeros that belong to group 1

	if !isempty(values)
		sort!(values; by=first) # sort them to get ranking
		first(first(values)) <= 0.0 && throw(DomainError("All non-zero values in matrix must be positive."))

		# First compute U=U₁ as if there were no zeros

		prev_value = NaN
		tie_count = 0 # current number of ties
		tie_count1 = 0 # current number of ties that belong to group 1

		for (rank,(v,b)) in enumerate(values)
			if v !== prev_value
				# We are ready to process the previous group of ties (e.g. up to rank-1)

				mean_rank_times2 = 2rank-tie_count-1
				Rtimes2 += mean_rank_times2*tie_count1
				tie_adjustment += tie_count*(tie_count^2 - 1)

				tie_count = tie_count1 = 0
			end

			prev_value = v
			tie_count += 1
			tie_count1 += b
			nz_count1 += b
		end
		# We are ready to process the final group of ties
		rank = length(values)+1
		mean_rank_times2 = 2rank-tie_count-1
		Rtimes2 += mean_rank_times2*tie_count1
		tie_adjustment += tie_count*(tie_count^2 - 1)
	end

	# Now adjust for zeros

	# 1. Offset U
	z_count = n1+n2-length(values)
	Rtimes2 += nz_count1*z_count*2 # each value added for group 1 above should have been z_count higher

	
	# 2. Add rank for zero-elements in group 1

	z_count1 = n1-nz_count1
	mean_zero_rank_times2 = z_count+1
	Rtimes2 += mean_zero_rank_times2*z_count1
	tie_adjustment += z_count*(z_count^2 - 1)

	Utimes2 = Rtimes2 - n1*(n1+1)
	return Utimes2/2, tie_adjustment
end

mannwhitney_σ(n1,n2,tie_adjustment) =
	sqrt(n1*n2/12 * (n1 + n2 + 1 - tie_adjustment/((n1+n2)*(n1+n2-1))))

function mannwhitney_single(X::AbstractSparseMatrix, j, groups, n1, n2)
	min(n1,n2)==0 && return 0.0, 0.0, 1.0 # degenerate case

	U, tie_adjustment = ustatistic_single(X, j, groups, n1, n2)

	m = n1*n2/2
	σ = mannwhitney_σ(n1,n2,tie_adjustment)

	# TODO: handle directional tests too
	d = U-m
	p = min(1, 2*ccdf(Normal(0,σ), abs(d)-0.5)) # 0.5 is the continuity correction factor
	z = σ>0 ? d/σ : 0.0 # standardized (signed) statistic - monotone with p, but never underflows
	return U, z, p
end


function mannwhitney_sparse(X::AbstractSparseMatrix, groups; kwargs...)
	@assert all(in((0,1,2)), groups)
	n1 = count(==(1), groups)
	n2 = count(==(2), groups)
	@assert n1>0
	@assert n2>0

	U = zeros(size(X,1))
	z = zeros(size(X,1))
	p = zeros(size(X,1))

	threaded_sparse_row_map(X; kwargs...) do Y, col, i
		U[i],z[i],p[i] = mannwhitney_single(Y,col,groups,n1,n2)
	end

	U, z, p
end


"""
	mannwhitney_resolve_groups(v, group_a, group_b=nothing) -> (group_a, group_b)

Resolve the two group labels for a Mann-Whitney U-test from the values `v`.

If `group_a` is not given (`nothing`), `v` must have exactly two unique values (ignoring
`missing`), which become the two groups. Returns concrete group labels - this is what should
be frozen when the test is later projected onto other data.
"""
function mannwhitney_resolve_groups(v, group_a=nothing, group_b=nothing)
	if group_a === nothing
		@assert group_b === nothing
		uv = unique(skipmissing(v))
		length(uv)==2 || throw(ArgumentError(string("Expected exactly two unique values, found: ", collect(uv), ".")))
		group_a, group_b = minmax(uv[1], uv[2])
	end
	group_a, group_b
end


"""
	mannwhitney_groups(v, group_a, group_b=nothing; h1_missing=:skip) -> Vector{Int}

Assign each element of `v` to a group for a Mann-Whitney U-test:
* `1` - equal to `group_a`.
* `2` - equal to `group_b`, or (if `group_b===nothing`) anything that is neither `group_a` nor `missing`.
* `0` - excluded (including `missing`).

If `h1_missing==:error`, an error is thrown if `v` contains any `missing` values.
"""
function mannwhitney_groups(v, group_a, group_b=nothing; h1_missing=:skip)
	@assert h1_missing in (:skip,:error)
	if h1_missing == :error && any(ismissing, v)
		throw(ArgumentError("Values contain missing, set `h1_missing=:skip` to skip them."))
	end

	maskA = isequal.(v, group_a)
	any(maskA) || throw(ArgumentError(string("Values don't contain group \"", group_a, "\".")))

	if group_b !== nothing
		maskB = isequal.(v, group_b)
		any(maskB) || throw(ArgumentError(string("Values don't contain group \"", group_b, "\".")))
	else
		maskB = .!isequal.(v, group_a) .& .!ismissing.(v)
		any(maskB) || throw(ArgumentError(string("Values only contain one group: \"", group_a, "\".")))
	end

	groups = zeros(Int, length(v))
	groups[maskA] .= 1
	groups[maskB] .= 2
	groups
end


"""
	mannwhitney_table2(matrix, var, groups; statistic_col="U", pvalue_col="pValue", z_col=nothing, do_sort=true, kwargs...)

Compute the Mann-Whitney U-test for each variable (row of the sparse `matrix`) given a
`groups` vector (see [`mannwhitney_groups`](@ref)), and return a copy of the `var` table with
the U statistics, z-scores and p-values added. `matrix` must be sparse.

The signed z-score `z = (U - n1*n2/2)/σ` is a standardized statistic that is monotone with the
p-value but never underflows; when `do_sort`, rows are sorted by `|z|` (most significant first)
whether or not the `z` column is included. Each of `statistic_col`/`pvalue_col`/`z_col` names an
output column, or omits it when set to `nothing` (`z` is omitted by default).
"""
function mannwhitney_table2(matrix, var, groups;
                            statistic_col="U", pvalue_col="pValue", z_col=nothing,
                            do_sort=true, kwargs...)
	U,z,p = mannwhitney_sparse(unblockify(matrix), groups; kwargs...)
	table = copy(var; copycols=do_sort)
	statistic_col !== nothing && insertcols!(table, statistic_col=>U; copycols=false)
	z_col !== nothing && insertcols!(table, z_col=>z; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	# Sort by |z| (two-sided significance) regardless of whether z is an output column.
	do_sort && (table = table[sortperm(z; by=abs, rev=true), :])
	table
end
