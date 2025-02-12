function logtransform_matrix(::Type{T}, X::SparseMatrixCSC; scale_factor) where T
	P,N = size(X)
	s = max.(1, sum(X; dims=1))
	nf = scale_factor ./ s

	# log( 1 + c*f/s)
	nzval = nonzeros(X)
	nzval_out = zeros(T, nnz(X))

	for j in 1:N
		irange = nzrange(X,j)
		# nzval_out[irange] .= log2.(1 .+ (@view nzval[irange]) .* nf[j])
		nzval_out[irange] .= convert.(T, log2.(1 .+ (@view nzval[irange]) .* nf[j]))
	end

	A = SparseMatrixCSC(P, N, copy(X.colptr), copy(X.rowval), nzval_out)
	MatrixRef(:A=>A)
end

function logtransform_matrix(::Type{T}, X::Matrix, scale_factor) where T
	s = max.(1, sum(X; dims=1))
	nf = scale_factor ./ s
	A = convert.(T, log2.(1 .+ X.*nf))
	MatrixRef(:A=>A)
end

logtransform_matrix(X; scale_factor) = logtransform_matrix(Float64, X; scale_factor)
