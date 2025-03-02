# TODO: this could also be implemented efficiently for MatrixExpressions - but only if f==identity
function var_counts_sum2(f::F, counts::SparseMatrixCSC{T}, ind) where {F,T}
	mask = falses(size(counts,1))
	mask[ind] .= true


	P,N = size(counts)
	@assert length(mask)==P
	f0 = f(zero(T))
	@assert iszero(f0) "Expected $f(0) to equal 0, got $f0."

	out = zeros(typeof(f0+f0), N) # Is there a better way to get the output type?

	# TODO: Can we make this a bit faster? If it wasn't for the mask, it would just be sum(f,counts;dims=1)
	R = rowvals(counts)
	V = nonzeros(counts)
	for j=1:N
		out[j] = sum(nzrange(counts,j); init=0) do k
			mask[R[k]] ? f(V[k]) : f0
		end
	end
	out	
end

