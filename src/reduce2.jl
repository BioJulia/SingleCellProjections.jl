struct SVDModel2 <: ProjectionModel
	U::Matrix{Float64}
	S::Vector{Float64}
end
SVDModel2(F::SVD) = SVDModel2(F.U, F.S)

function svd_project(model::SVDModel2, X)
	U = model.U
	S = model.S
	V = X'U # TODO: compute F.U'X instead to get Vt directly
	V ./= max.(S,1e-100)' # To avoid NaNs if any singular value is zero
	SVD(U,S,Matrix(V'))
end
