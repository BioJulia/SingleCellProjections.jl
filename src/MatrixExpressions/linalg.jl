LinearAlgebra.adjoint(X::MatrixRef) = MatrixRef(X.name, adjoint(X.matrix))
function LinearAlgebra.adjoint(X::MatrixProduct)
	f = reverse(X.factors) # copies vector and keeps eltype
	MatrixProduct(map!(adjoint,f,f))
end
LinearAlgebra.adjoint(X::MatrixSum) = MatrixSum(adjoint.(X.terms))

Base.:*(A::MatrixRef, X::AbstractVecOrMat) = A.matrix*X
Base.:*(X::AbstractVecOrMat, A::MatrixRef) = X*A.matrix

Base.:*(A::MatrixExpression, X::AbstractVecOrMat) = compute(matrixproduct(A, :X=>X))
Base.:*(X::AbstractVecOrMat, A::MatrixExpression) = compute(matrixproduct(:X=>X ,A))


# ------------------------------------------------------------------------------
# First implementation of computations.
# Needs matrix chain optimization and other optimizations.


function compute(A::MatrixProduct)
	i = findfirst(x->x isa MatrixSum, A.factors)
	if i===nothing
		adjoint_sparse_chain_mul(A)
	else
		# distribute factors over sum at index i
		s = A.factors[i]::MatrixSum

		pre = A.factors[1:i-1]
		post = A.factors[i+1:end]

		products = [matrixproduct(pre...,term,post...) for term in s.terms]
		compute(MatrixSum(products))
	end
end


compute(A::MatrixSum) = sum(compute, A.terms)


# Should be fast in Julia 1.7+ for different matrix types - TEST!
# But we probably want to support Julia 1.6. So maybe some additional methods are needed.
compute_diaggram(A::AbstractMatrix) = vec(sum(abs2, A; dims=1))




compute(D::DiagGram{<:MatrixRef}) = compute_diaggram(D.A.matrix)
function compute(D::DiagGram{<:MatrixProduct})
	diaggram_chain_mul(D.A)
end

function compute(D::DiagGram{MatrixSum})
	s = sum(Ai->compute(DiagGram(Ai)), D.A.terms) # DiagGram(AᵢᵀAᵢ)
	# iterate upper triangle
	for j=2:length(D.A.terms)
		for i=1:j-1
			s .+= 2 .* compute(Diag(matrixproduct(D.A.terms[i]',D.A.terms[j]))) # 2*diag(AᵢᵀAⱼ)
		end
	end
	s
end


# TODO: do not assume Float64
compute_diagmul(A,B) = Float64[dot(view(A,i,:), view(B,:,i)) for i=1:size(A,1)] # generic fallback
# compute_diagmul(A::Adjoint{<:Any,<:AbstractSparseMatrix},B::AbstractSparseMatrix) =
# 	Float64[dot(view(A.parent,:,i), view(B,:,i)) for i=1:size(A,1)]
compute_diagmul(A::Adjoint,B) =
	Float64[dot(view(A.parent,:,i), view(B,:,i)) for i=1:size(A,1)] # TODO: thread?

# TODO: Not currently used by diag_chain_mul, we might want to enable it?
# function compute_diagmul(A::AbstractSparseMatrix, B::Adjoint{<:Any,<:AbstractSparseMatrix})
# 	out = zeros(size(A,1))

# 	AR = rowvals(A)
# 	BR = rowvals(B.parent)
# 	AV = nonzeros(A)
# 	BV = nonzeros(B.parent)

# 	for j in 1:size(A,2) # TODO: Thread here!
# 		Anzr = nzrange(A,j)
# 		Bnzr = nzrange(B.parent,j)

# 		(length(Anzr)==0 || length(Bnzr)==0) && continue

# 		Ak = 1
# 		Bk = 1
# 		Ai = AR[Anzr[1]]
# 		Bi = BR[Bnzr[1]]

# 		@inbounds while true
# 			if Ai==Bi
# 				out[Ai] += AV[Anzr[Ak]] * BV[Bnzr[Bk]]
# 				Ak += 1
# 				Bk += 1
# 				Ak>length(Anzr) && break
# 				Bk>length(Bnzr) && break
# 				Ai = AR[Anzr[Ak]]
# 				Bi = BR[Bnzr[Bk]]
# 			elseif Ai<Bi
# 				Ak += 1
# 				Ak>length(Anzr) && break
# 				Ai = AR[Anzr[Ak]]
# 			else
# 				Bk += 1
# 				Bk>length(Bnzr) && break
# 				Bi = BR[Bnzr[Bk]]
# 			end
# 		end
# 	end
# 	out
# end


# function compute(D::Diag)
# 	@assert !isempty(D.A.factors)
# 	if length(D.A.factors) == 1
# 		diag(D.A.factors[1].matrix)
# 	else
# 		diag_chain_mul(D.A)
# 	end
# end

function compute(D::Diag)
	@assert !isempty(D.A.factors)

	ind = findfirst(x->x isa MatrixSum, D.A.factors)
	if ind===nothing
		if length(D.A.factors) == 1
			diag(D.A.factors[1].matrix)
		else
			diag_chain_mul(D.A)
		end
	else
		# Distribute over the sum
		S = D.A.factors[ind]::MatrixSum

		out = zeros(size(D.A,1))
		for t in S.terms
			out .+= compute(Diag(matrixproduct(D.A.factors[1:ind-1]..., t, D.A.factors[ind+1:end]...)))
		end
		out
	end
end
