function orthonormal_design2(X, Q0=nothing; rtol=sqrt(eps()))
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


function _linear_test(A, ssA, h1, h0=nothing)
	@assert size(A,2) == size(h1,1)
	@assert h0 === nothing || size(A,2) == size(h0,1)

	# `h0 === nothing` means there is no null model (not even an intercept).
	Q0 = h0 === nothing ? nothing : orthonormal_design2(h0)[1]
	Q1,scale = orthonormal_design2(h1, Q0) # orthonormal_design2 skips orthogonalization when Q0===nothing

	# fit models
	β1 = A*Q1

	# `ssA` (the per-variable sum of squares of A) is computed as a separate cached job and passed in.

	ssβ0 = Q0 === nothing ? 0.0 : vec(sum(abs2, A*Q0; dims=2))
	ssβ1 = vec(sum(abs2, β1; dims=2))

	ssExplained = max.(0.0, ssβ1)
	ssUnexplained = max.(0.0, ssA .- ssβ1 .- ssβ0)
	rank0 = Q0 === nothing ? 0 : size(Q0,2)
	rank1 = size(Q1,2)+rank0

	ssExplained, ssUnexplained, rank0, rank1, β1, scale
end



function ftest_table(matrix, var::DataFrame, ssA, h1, h0=nothing;
                     statistic_col=nothing, pvalue_col=nothing,
                     do_sort=true)
	ssExplained, ssUnexplained, rank0, rank1, _, _ = _linear_test(matrix, ssA, h1, h0)
	N = size(matrix,2)
	ν1 = (rank1-rank0)
	ν2 = (N-rank1)

	if ν1>0 && ν2>0
		F = max.(0.0, (ν2/ν1) * ssExplained./ssUnexplained)
		p = ccdf.(FDist(ν1,ν2), F)
	else
		F = zeros(size(ssExplained))
		p = ones(size(ssExplained))
	end

	table = copy(var; copycols=do_sort)
	statistic_col !== nothing && insertcols!(table, statistic_col=>F; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)

	# Sort by F (larger = more significant) regardless of whether the F column is added.
	do_sort && permute!(table, sortperm(F; rev=true))

	table
end


function ttest_table(matrix, var, ssA, h1, h1_scale, h0=nothing;
                     statistic_col=nothing, pvalue_col=nothing, difference_col=nothing,
                     do_sort=true)
	_, ssUnexplained, rank0, rank1, β1, scale = _linear_test(matrix, ssA, h1, h0)
	N = size(matrix,2)
	ν1 = (rank1-rank0)
	ν2 = (N-rank1)

	if ν1==1 && ν2>0
		t = vec(β1./sqrt.(max.(0.0,(ν1/ν2).*ssUnexplained)))
		p = min.(1.0, 2.0.*ccdf.(TDist(ν2), abs.(t)))
		# d = vec(β1)/(scale*_covariate_scale(only(h1.covariates)))
		d = vec(β1)/(scale*h1_scale)
	else
		t = zeros(size(ssUnexplained))
		p = ones(size(ssUnexplained))
		d = zeros(size(ssUnexplained))
	end

	table = copy(var; copycols=do_sort)
	statistic_col !== nothing && insertcols!(table, statistic_col=>t; copycols=false)
	pvalue_col !== nothing && insertcols!(table, pvalue_col=>p; copycols=false)
	difference_col !== nothing && insertcols!(table, difference_col=>d; copycols=false)

	# Sort by |t| (larger = more significant) regardless of whether the t column is added.
	do_sort && permute!(table, sortperm(t; by=abs, rev=true))

	table
end
