function svd_projected_svt(U, X)
	ΣV = X'U # TODO: compute U'X instead to get Vt directly
	Matrix(ΣV')
end
svd_projected_svt(U::ReadOnlyArray, X) = svd_projected_svt(parent(U), X)

svd_project_mul_sinv(ΣVt, S) = ΣVt ./ max.(S,1e-100) # To avoid NaNs if any singular value is zero

# # Should we have this? Not used by Jobs.
# function svd_project(F::SVD, X)
# 	Vt = svd_projected_vt(F.U, F.S, X)
# 	SVD(F.U, F.S, Vt)
# end
