# TEMP simplified normalization for testing projections via Specs
struct CenteringModel2 <: ProjectionModel
	negβT::Matrix{Float64}

	function CenteringModel2(A)
		N = size(A,2)
		X = ones(N,1)
		negβT = (A*X) .* (-1/N)
		new(negβT)
	end
end

function center_matrix_project(model::CenteringModel2, A)
	N = size(A,2)
	X = ones(N,1)
	matrixsum(_named_matrix(A,:A), matrixproduct(Symbol("(-β)")=>model.negβT, :X=>X'))
end


# Normalization without model struct
function negative_regression_matrix(matrix, dm::Matrix; rtol=sqrt(eps()))
	A = matrix
	X = dm

	# TODO: No need to run svd etc. if there just is an intercept.
	F = svd(X)
	negΣinv = Diagonal([σ>rtol ? -1.0/σ : 0.0 for σ in F.S]) # cutoff for numerical stability
	AU = A*F.U
	(AU*negΣinv)*F.Vt
end

# # TODO: Replace this with direct calls to matrixsum and matrixproduct?
# normalize_matrix2(matrix, negβT::Matrix, dm::Matrix) =
# 	matrixsum(_named_matrix(matrix,:A), matrixproduct(Symbol("(-β)")=>negβT, :X=>dm'))
