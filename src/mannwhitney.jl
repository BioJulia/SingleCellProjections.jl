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
				# We are ready to process the last group of ties (e.g. up to rank-1)

				# range = rank-tie_count:rank-1
				# mean_rank = (rank-tie_count + rank-1)/2

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

	# range: 1:z_count
	# mean_rank = (1+z_count)/2

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
	min(n1,n2)==0 && return 0.0, 1.0 # degenerate case
	U, tie_adjustment = ustatistic_single(X, j, groups, n1, n2)

	m = n1*n2/2
	σ = mannwhitney_σ(n1,n2,tie_adjustment)

	# TODO: handle directional tests too
	z = U-m
	p = min(1, 2*ccdf(Normal(0,σ), abs(z)-0.5)) # 0.5 is the continuity correction factor
	return U, p
end


function mannwhitney_sparse(X::AbstractSparseMatrix, groups; kwargs...)
	@assert all(in((0,1,2)), groups)
	n1 = count(==(1), groups)
	n2 = count(==(2), groups)
	@assert n1>0
	@assert n2>0

	U = zeros(size(X,1))
	p = zeros(size(X,1))

	threaded_sparse_row_map(X; kwargs...) do Y, col, i
		U[i],p[i] = mannwhitney_single(Y,col,groups,n1,n2)
	end

	U, p
end
