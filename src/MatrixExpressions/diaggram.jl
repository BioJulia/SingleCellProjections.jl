_diaggram_cost(::Type{<:AdjOrDense}, A) = prod(size(A))
_diaggram_cost(::Type{<:AdjOrSparse}, A) = prod(size(A))*A.pnz*SPARSE_TIME_FACTOR

_diaggram_cost(A::MatrixInfo) = _diaggram_cost(A.type, A)

function diaggram_chain_mul(A::MatrixProduct)::Vector
	# Compare
	# evaluating A followed by sum(abs2,A;dims=1)
	resultsA, orderA = plan_adjoint_sparse_chain(A)
	# to evaluating diag(AᵀA)
	_, orderD = plan_diag_chain(matrixproduct(A',A)) # Ideally, we should avoid recomputing partial results that appear twice

	costA = orderA.cost + _diaggram_cost(getsubresult(resultsA[1,end],false).matrixinfo)

	# @show costA, orderD.cost

	if costA <= orderD.cost
		# @info "sum(abs2(A))"
		X = apply_chain(orderA)
		compute_diaggram(X)
	else
		# @info "diag(AᵀA)"
		apply_chain(orderD)
	end
end
