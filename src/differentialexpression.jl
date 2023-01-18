function _group_sum(X::SparseMatrixCSC, obs_ind)
	P = size(X,1)
	V = nonzeros(X)
	R = rowvals(X)
	s = zeros(P)
	for j in obs_ind
		for k in nzrange(X,j)
			i = R[k]
			v = V[k]

			s[i] += v
		end
	end
	s
end

function _group_square_sum(X::SparseMatrixCSC, m, obs_ind)
	P = size(X,1)
	V = nonzeros(X)
	R = rowvals(X)

	s2 = zeros(P)
	nvalues = zeros(Int,P)
	for j in obs_ind
		for k in nzrange(X,j)
			i = R[k]
			v = V[k]

			s2[i] += (v-m[i])^2
			nvalues[i] += 1
		end
	end

	s2 += m.^2 .* (length(obs_ind) .- nvalues) # include all structural zeros

	s2
end


"""
	differentialexpression(data::DataMatrix{<:SparseMatrixCSC}, groups::AbstractVector)

Assumes `data` is logtransformed.
`groups` should be a vector with 1 for observations in group 1, 2 for observations in group 2 and
`missing` for other observations.
"""
function differentialexpression(data::DataMatrix{<:SparseMatrixCSC}, groups::AbstractVector)
	@assert all(in((1,2)), skipmissing(groups)) "There must be exactly two groups (use missing values to skip observations)"
	g1_ind = findall(x->x===1, groups)
	g2_ind = findall(x->x===2, groups)
	@assert length(g1_ind)>1 "Group 1 must have at least two observations"
	@assert length(g2_ind)>1 "Group 2 must have at least two observations"

	X = data.matrix
	P,N = size(data.matrix)


	n1 = length(g1_ind)
	n2 = length(g2_ind)
	ntotal = n1 + n2 # actual number of used cells
	dfUnexplained = ntotal-2


	gs1 = _group_sum(X, g1_ind)
	gs2 = _group_sum(X, g2_ind)

	m1 = gs1./n1
	m2 = gs2./n2

	s2_1 = _group_square_sum(X, m1, g1_ind)
	s2_2 = _group_square_sum(X, m2, g2_ind)

	s = sqrt.( (s2_1.+s2_2).*(ntotal/(float(dfUnexplained)*n1*n2)) )


	log_fold_change = m2.-m1
	mean_expression = (gs1.+gs2)./ntotal
	t_statistic = log_fold_change.*min.(floatmax(),1.0./s) # avoid NaNs - set t_statistic to 0
	p_value = 2 .* ccdf.(TDist(dfUnexplained), abs.(t_statistic))

	df = DataFrame(;t_statistic, p_value, log_fold_change, mean_expression)
	hcat(data.var, df)
end



# groups are 1 for group1 and 2 for group2. missings are skipped.
# deprecated.
function differentialexpression_old(counts::SparseMatrixCSC{<:Integer,<:Integer}, groups::AbstractVector{}; scale_factor=1e4)
	counts = copy(counts') # HACK because the old code was written for transposed matrices


	N,P = size(counts)
	@assert extrema(skipmissing(groups))==(1,2)

	cellSums = sum(counts; dims=2)
	cellNormFactor = scale_factor ./ cellSums

	groupCount1 = groupCount2 = 0 # number of cells in each group
	for g in skipmissing(groups)
		if g==1
			groupCount1 += 1
		else
			groupCount2 += 1
		end
	end
	N2 = groupCount1 + groupCount2 # actual number of used cells
	factor = sqrt(groupCount1*groupCount2/N2)
	dfUnexplained = N2-2


	# outputs
	meanLogCPM = zeros(P)
	logFC = zeros(P)
	tStatistic = zeros(P)
	pValue = zeros(P)

	countVals = nonzeros(counts)
	rows = rowvals(counts)

	scratchLogCPM = zeros(maximum(length.(nzrange.(Ref(counts),1:P)))) # as many values as the longest column

	for j=1:P # for each gene
		groupSum1 = groupSum2 = 0.0
		for (c,k) in enumerate(nzrange(counts,j))
			i = rows[k]
			g = groups[i]
			ismissing(g) && continue # skipped cell
			g::Int
			v = log2(1.0 + countVals[k]*cellNormFactor[i])
			scratchLogCPM[c] = v
			if g==1
				groupSum1 += v
			else
				groupSum2 += v
			end
		end

		meanLogCPM[j] = (groupSum1+groupSum2)/N2
		logFC[j] = groupSum2/groupCount2 - groupSum1/groupCount1

		explained = factor * logFC[j]
		total2 = 0.0
		for (c,k) in enumerate(nzrange(counts,j))
			i = rows[k]
			g = groups[i]
			ismissing(g) && continue # skipped cell
			v = scratchLogCPM[c]
			total2 += v*(v-2meanLogCPM[j])
		end

		total2 += N2*meanLogCPM[j]^2 # ∑(x-x̄)² = ∑x² - 2∑xx̄ + ∑x̄². The first two terms are computed in the sparse loop and the final term here.
		unexplained2 = max(0.0, total2 - explained^2) # avoid rounding errors causing negative values

		tStatistic[j] = explained * min(floatmax(), sqrt(dfUnexplained/unexplained2)) # avoid NaNs by setting tStatistic to 0 if explained=0 and unexplained2=0.
		pValue[j] = 2*ccdf(TDist(dfUnexplained), abs(tStatistic[j]))
	end

	tStatistic,pValue,logFC,meanLogCPM
end
