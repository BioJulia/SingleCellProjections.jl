function svd_projected_vt(U, S, X)
	V = X'U # TODO: compute U'X instead to get Vt directly
	V ./= max.(S,1e-100)' # To avoid NaNs if any singular value is zero
	Matrix(V')
end

# Should we have this? Not used by Jobs.
function svd_project(F::SVD, X)
	Vt = svd_projected_vt(F.U, F.S, X)
	SVD(F.U, F.S, Vt)
end
