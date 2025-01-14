# TEMP simplified normalization for testing projections via Specs
struct CenteringModel2 <: ProjectionModel
	negβT::Matrix{Float64}
end
function CenteringModel2(A)
	N = size(A,2)
	X = ones(N,1)
	negβT = (A*X) .* (-1/N)
	CenteringModel2(negβT)
end

function project2(model::CenteringModel2, A)
	N = size(A,2)
	X = ones(N,1)
	matrixsum(_named_matrix(A,:A), matrixproduct(Symbol("(-β)")=>model.negβT, :X=>X'))
end
