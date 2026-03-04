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
	# factors::Vector{Union{MatrixRef,T}}
	factors::Any # Refactoring TODO: Find a better solution (we need to handle deduplication somehow, and that can narrow the type)

	function MatrixProduct{T}(factors) where T
		# @show eltype(factors) == Union{MatrixRef,T} # DEBUG

		@assert !isempty(factors)
		sz_prev = (0,size(first(factors),1))
		for (i,f) in enumerate(factors)
			sz_i = size(f)
			sz_i[1] != sz_prev[2] && throw(DimensionMismatch("factors[$(i-1)] has size $sz_prev, factors[$i] has size $sz_i."))
			sz_prev = sz_i
		end
		new{T}(factors)
	end
end
struct MatrixSum <: AbstractMatrixSum
	# terms::Vector{Union{MatrixRef,MatrixProduct}}
	terms::Any # Refactoring TODO: Find a better solution (we need to handle deduplication somehow, and that can narrow the type)

	function MatrixSum(terms)
		# @show eltype(terms) == Union{MatrixRef,MatrixProduct} # DEBUG

		@assert !isempty(terms)
		sz = size(first(terms))
		for (i,t) in enumerate(terms)
			sz_i = size(t)
			sz_i != sz && throw(DimensionMismatch("terms[1] has size $sz, terms[$i] has size $sz_i."))
		end
		new(terms)
	end
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
matrixexpression((_,X)::Pair{Symbol,<:MatrixExpression}) = X # ignore name - we cannot name an entire expression, the parts already have names
matrixexpression(X) = MatrixRef(X)



_pushfactors!(factors, X) = push!(factors, matrixexpression(X))
_pushfactors!(factors, X::MatrixProduct) = append!(factors, X.factors)
_pushfactors!(factors, (_,X)::Pair{Symbol,<:MatrixProduct}) = append!(terms, X.factors) # ignore name - we cannot name an entire expression, the parts already have names

function matrixproduct(args...)
	factors = Union{MatrixRef,MatrixSum}[]
	for x in args
		_pushfactors!(factors, x)
	end
	MatrixProduct(factors)
end


_pushterms!(terms, X) = push!(terms, matrixexpression(X))
_pushterms!(terms, X::MatrixSum) = append!(terms, X.terms)
_pushterms!(terms, (_,X)::Pair{Symbol,<:MatrixSum}) = append!(terms, X.terms) # ignore name - we cannot name an entire expression, the parts already have names

function matrixsum(args...)
	terms = Union{MatrixRef,MatrixProduct}[]
	for x in args
		_pushterms!(terms, x)
	end
	MatrixSum(terms)
end
