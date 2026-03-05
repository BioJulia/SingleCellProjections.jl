# TODO: this could also be implemented efficiently for MatrixExpressions - but only if f==identity
function _masked_column_sum!(f::F, out, counts::SparseMatrixCSC{T}, mask, f0) where {F,T}
	# TODO: Can we make this a bit faster? If it wasn't for the mask, it would just be sum(f,counts;dims=1)
	R = rowvals(counts)
	V = nonzeros(counts)
	for j in 1:size(counts,2)
		out[j] = sum(nzrange(counts,j); init=0) do k
			mask[R[k]] ? f(V[k]) : f0
		end
	end
	out
end


# TODO: this could also be implemented efficiently for MatrixExpressions - but only if f==identity
function _masked_row_sum!(f::F, out, counts::SparseMatrixCSC{T}, mask, f0) where {F,T}
	# TODO: Can we make this a bit faster? If it wasn't for the mask, it would just be sum(f,counts;dims=2)
	R = rowvals(counts)
	V = nonzeros(counts)
	for j in 1:size(counts,2)
		for k in nzrange(counts,j)
			i = R[k]
			out[i] += mask[j] ? f(V[k]) : f0
		end
	end
	out
end


"""
	counts_sum(f, counts, ind; dims)

See also: [`counts_fraction`](@ref)
"""
function counts_sum(f::F, counts::SparseMatrixCSC{T}, ind; dims::Integer) where {F,T}
	@assert dims in (1,2)

	f0 = f(zero(T))
	@assert iszero(f0) "Expected $f(0) to equal 0, got $f0."

	mask = falses(size(counts,dims))
	mask[ind] .= true

	if dims == 1
		out = zeros(typeof(f0+f0), size(counts,2)) # Is there a better way to get the output type?
		_masked_column_sum!(f, out, counts, mask, f0)
	else#if dims == 2
		out = zeros(typeof(f0+f0), size(counts,1)) # Is there a better way to get the output type?
		_masked_row_sum!(f, out, counts, mask, f0)
	end
	out
end
