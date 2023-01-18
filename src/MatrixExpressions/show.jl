function Base.show(io::IO, X::MatrixRef)
	if get(io,:compact,false)
		print(io, X.name, X.matrix isa Adjoint ? "'" : "")
		# print(io, X.name)
	else
		print(io, X.name, "::", typeof(X.matrix), size(X.matrix))
	end
end
function Base.show(io::IO, A::MatrixProduct)
	io2 = IOContext(io, :compact=>true)
	for X in A.factors
		s = X isa MatrixSum
		s && print(io2, '(')
		print(io2, X)
		s && print(io2, ')')
	end
end
function Base.show(io::IO, A::MatrixSum)
	io2 = IOContext(io, :compact=>true)
	join(io2, A.terms, '+')
end
