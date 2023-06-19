"""
	ustatistic_single(X, j, groups, n1, n2)

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

	sort!(values; by=first) # sort them to get ranking

	# First compute U=U₁ as if there were no zeros
	
	Rtimes2 = 0 # Due to ties, possible values are of the form k/2. We thus store U*2 here, to be able to work with integers.

	prev_value = NaN
	tie_count = 0 # current number of ties
	tie_count1 = 0 # current number of ties that belong to group 1
	nz_count1 = 0 # total number of non-zeros that belong to group 1
	
	for (rank,(v,b)) in enumerate(values)
		if v !== prev_value
			# We are ready to process the last group of ties (e.g. up to rank-1)

			# range = rank-tie_count:rank-1
			# mean_rank = (rank-tie_count + rank-1)/2

			mean_rank_times2 = 2rank-tie_count-1
			Rtimes2 += mean_rank_times2*tie_count1

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

	Utimes2 = Rtimes2 - n1*(n1+1)
	Utimes2/2
end
