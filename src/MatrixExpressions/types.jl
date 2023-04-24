abstract type MatrixExpression end

# This type has the sole purpose of allowing circular references below, i.e. a MatrixProduct
# can contain a MatrixSum that can contain a MatrixProduct.
abstract type AbstractMatrixSum <: MatrixExpression end

struct MatrixRef{T} <: MatrixExpression
	name::Symbol
	matrix::T
end
MatrixRef(matrix) = MatrixRef(Symbol("(",String(rand('a':'z',6)),")"), matrix)
MatrixRef(t::Tuple{Symbol,T}) where T = MatrixRef(t...)
MatrixRef(p::Pair{Symbol,T}) where T = MatrixRef(p...)

struct MatrixProduct{T<:AbstractMatrixSum} <: MatrixExpression
	factors::Vector{Union{MatrixRef,T}}
end
struct MatrixSum <: AbstractMatrixSum
	terms::Vector{Union{MatrixRef,MatrixProduct}}
end

MatrixProduct(factors) = MatrixProduct{MatrixSum}(factors)



struct DiagGram{T<:Union{MatrixRef,MatrixSum,MatrixProduct}}
	A::T
end
struct Diag
	A::MatrixProduct
end

Base.size(A::MatrixRef) = size(A.matrix)
Base.size(A::MatrixProduct) = (size(A.factors[1],1), size(A.factors[end],2))
Base.size(A::MatrixSum) = size(A.terms[1])

Base.size(A::MatrixExpression, ind) = ind>2 ? 1 : size(A)[ind]

# Do we want this? Or solve somewhat differently?
Base.eltype(::MatrixExpression) = Float64


Base.:(==)(A::MatrixRef, B::MatrixRef) = A.name == B.name && A.matrix == B.matrix
Base.:(==)(A::MatrixProduct, B::MatrixProduct) = A.factors == B.factors
Base.:(==)(A::MatrixSum, B::MatrixSum) = A.terms == B.terms
Base.:(==)(A::DiagGram, B::DiagGram) = A.A == B.A
Base.:(==)(A::Diag, B::Diag) = A.A == B.A


function _copy_rec(v::Vector)
	out = similar(v)
	for i in eachindex(v)
		out[i] = copy(v[i])
	end
	out
end

Base.copy(A::MatrixRef) = MatrixRef(A.name, A.matrix)
Base.copy(A::MatrixProduct{T}) where T = MatrixProduct{T}(_copy_rec(A.factors))
Base.copy(A::MatrixSum) = MatrixSum(_copy_rec(A.terms))
Base.copy(A::DiagGram) = DiagGram(copy(A.A))
Base.copy(A::Diag) = Diag(copy(A.A))




matrixexpression(X::MatrixExpression) = X
matrixexpression(X) = MatrixRef(X)


_pushfactors!(A::MatrixProduct, X) = push!(A.factors, matrixexpression(X))
_pushfactors!(A::MatrixProduct, X::MatrixProduct) = append!(A.factors,X.factors)

function matrixproduct(args...)
	A = MatrixProduct([])
	for x in args
		_pushfactors!(A,x)
	end
	A
end

_pushterms!(A::MatrixSum, X) = push!(A.terms, matrixexpression(X))
_pushterms!(A::MatrixSum, X::MatrixSum) = append!(A.terms,X.terms)

function matrixsum(args...)
	A = MatrixSum([])
	for x in args
		_pushterms!(A,x)
	end
	A
end
