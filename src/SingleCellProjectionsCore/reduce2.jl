function svd_project(F::SVD, X)
	U = F.U
	S = F.S
	V = X'U # TODO: compute F.U'X instead to get Vt directly
	V ./= max.(S,1e-100)' # To avoid NaNs if any singular value is zero
	SVD(U,S,Matrix(V'))
end
