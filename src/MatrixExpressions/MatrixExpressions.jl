module MatrixExpressions

export
	MatrixExpression,
	MatrixRef,
	MatrixProduct,
	MatrixSum,
	DiagGram,
	Diag,
	matrixexpression,
	matrixproduct,
	matrixsum,
	compute

using LinearAlgebra
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC

import AbstractTrees # for pretty printing

include("types.jl")
include("show.jl")
include("chain.jl")
include("basic_chain.jl") # Will perhaps only be an example later, not actually used.
include("adjoint_sparse_chain.jl")
include("diagmul_chain.jl")
include("diaggram.jl")
include("linalg.jl")

end
