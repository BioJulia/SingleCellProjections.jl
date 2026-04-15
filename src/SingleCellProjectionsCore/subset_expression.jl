# can we find a better name?
index_isnoop(I::Colon, ::Int) = true
index_isnoop(I::AbstractVector{<:Bool}, n::Int) = length(I)==n && all(I)
index_isnoop(I::AbstractVector{<:Integer}, n::Int) = I == 1:n
index_isnoop(::Any, ::Int) = false


_subsetmatrix(A::MatrixSum, I::Index, J::Index) =
	MatrixSum(_subsetmatrix.(A.terms, Ref(I), Ref(J)))

function _subsetmatrix(A::MatrixProduct, I::Index, J::Index)
	@assert length(A.factors)>=2 # Otherwise it shouldn't be a product
	factors = vcat(_subsetmatrix(first(A.factors), I, :), @view(A.factors[2:end-1]), _subsetmatrix(last(A.factors), :, J))
	MatrixProduct(factors)
end

function _subsetted_name(A::MatrixRef{T}, I::Index, J::Index) where T
	left = !index_isnoop(I, size(A,1))
	right = !index_isnoop(J, size(A,2))

	MatrixExpressions._is_adjoint_type(T) && ((left,right) = (right,left))

	left && right && return Symbol(A.name,'ₛ')
	left && return Symbol(A.name,'ₗ')
	right && return Symbol(A.name,'ᵣ')
	A.name
end


function _subsetmatrix(A::MatrixRef, I::Index, J::Index)
	if index_isnoop(I, size(A,1)) && index_isnoop(J, size(A,2))
		A
	else
		MatrixRef(_subsetted_name(A,I,J), _subsetmatrix(A.matrix,I,J))
	end
end
