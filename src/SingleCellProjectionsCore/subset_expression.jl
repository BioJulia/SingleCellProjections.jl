# TODO: Move this functionality into MatrixExpressions?

const AdjOrTransSparse = Union{Adjoint{<:Any,<:AbstractSparseMatrix{Tv,Ti}},
                               Transpose{<:Any,<:AbstractSparseMatrix{Tv,Ti}}} where {Tv,Ti}

_name(left::Val{true}, A) = Symbol(A.name,'ₗ')
_name(left::Val{false}, A) = Symbol(A.name,'ᵣ')

# sz is the size of matrix to subset
_ind2sparse(left::Val{true},  I, sz) = MatrixRef(:Sₗ, sparse(1:length(I),I,true,length(I),sz[1]))
_ind2sparse(left::Val{false}, I, sz) = MatrixRef(:Sᵣ, sparse(I,1:length(I),true,sz[2],length(I)))

ind2sparse(args...) = _ind2sparse(args...)

_length_check(left::Val{true},  I, sz) = @assert length(I) == sz[1]
_length_check(left::Val{false}, I, sz) = @assert length(I) == sz[2]
function ind2sparse(left, I::AbstractVector{Bool}, sz)
	_length_check(left, I, sz)
	_ind2sparse(left, findall(I), sz)
end


nselected(I) = length(I)
nselected(I::AbstractVector{Bool}) = count(I)

fractionselected(left::Val{true},  A, I) = nselected(I) / size(A,1)
fractionselected(left::Val{false}, A, I) = nselected(I) / size(A,2)


split_first(left::Val{true}, v) = first(v), @view v[2:end]
split_first(left::Val{false}, v) = last(v), @view v[1:end-1]

_product(left::Val{true}, args...) = matrixproduct(args...)
_product(left::Val{false}, args...) = matrixproduct(reverse(args)...)


_orderfactors(left::Val{true}, v) = v
_orderfactors(left::Val{false}, v) = reverse(v)

function _subset(left, A::MatrixProduct, I)
	@assert !isempty(A.factors)
	remaining = A.factors
	prefixes = similar(remaining, 0)

	F1, remaining = split_first(left, remaining)
	while !isempty(remaining)
		F1 isa MatrixRef{<:Diagonal} || break

		D = MatrixRef(_name(left,F1), Diagonal(F1.matrix.diag[I]))
		push!(prefixes, D)
		F1, remaining = split_first(left, remaining)
	end
	_product(left, prefixes..., _subset(left,F1,I), _orderfactors(left,remaining)...)
end


_subset(left, A::MatrixSum, I) = MatrixSum(_subset.(left, A.terms, Ref(I)))


_subset_ref(left::Val{true}, A::MatrixRef, I) = MatrixRef(_name(left,A), A.matrix[I,:])
_subset_ref(left::Val{false}, A::MatrixRef, I) = MatrixRef(_name(left,A), A.matrix[:,I])

_subset_ref(left::Val{true}, A::MatrixRef{<:Adjoint}, I) = _subset_ref(Val(false), A', I)'
_subset_ref(left::Val{false}, A::MatrixRef{<:Adjoint}, I) = _subset_ref(Val(true), A', I)'

_subset_ref(left::Val{true}, A::MatrixRef{<:Transpose}, I) = transpose(_subset_ref(Val(false), transpose(A), I))
_subset_ref(left::Val{false}, A::MatrixRef{<:Transpose}, I) = transpose(_subset_ref(Val(true), transpose(A), I))


function _subset(left, D::MatrixRef{<:Diagonal}, I) # silly edge case
	D2 = MatrixRef(_name(left,D), Diagonal(D.matrix.diag[I]))
	S = ind2sparse(left,I,size(D))
	_product(left,D2,S)
end


function _bytesize(X::AbstractSparseMatrix{Tv,Ti}) where {Tv,Ti}
	nnz(X)*(sizeof(Tv)+sizeof(Ti)) + size(X,2)*sizeof(Ti)
end
_bytesize(X::AdjOrTransSparse) = _bytesize(X.parent)
_bytesize(X) = prod(size(X))*sizeof(eltype(X)) # fallback assuming X is dense


function _subset(left, A::MatrixRef, I)
	# Heuristics to choose behavior
	if fractionselected(left, A, I) <= 1.0    # Never subset if we would get a larger matrix out
		if _bytesize(A.matrix) < 10^8 ||      # The original matrix is small enough
		   fractionselected(left, A, I) < 0.1 # The result is much smaller than the original

			# actually subset
			return _subset_ref(left, A, I)
		end
	end

	# insert a sparse matrix that does the subsetting
	S = ind2sparse(left,I,size(A))
	return _product(left, S, A)
end


# can we find a better name?
index_isnoop(I::Colon, ::Int) = true
index_isnoop(I::AbstractVector{<:Bool}, n::Int) = length(I)==n && all(I)
index_isnoop(I::AbstractVector{<:Integer}, n::Int) = I == 1:n
index_isnoop(::Any, ::Int) = false

function _subsetmatrix(X::MatrixExpression, I::Index, J::Index)
	if !index_isnoop(I, size(X,1))
		X = _subset(Val(true), X, I)
	end
	if !index_isnoop(J, size(X,2))
		X = _subset(Val(false), X, J)
	end
	X
end
