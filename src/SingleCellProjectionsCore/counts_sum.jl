# TODO: this could also be implemented efficiently for MatrixExpressions - but only if f==identity
function _masked_column_sum!(f::F, out, counts::SparseMatrixCSC{T}, mask, f0) where {F,T}
	# TODO: Can we make this a bit faster? If it wasn't for the mask, it would just be sum(f,counts;dims=1)
	R = rowvals(counts)
	V = nonzeros(counts)
	for j in 1:size(counts,2)
		out[j] += sum(nzrange(counts,j); init=0) do k
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


function _masked_column_sum!(f, out, A::Blocks, mask, f0)
	row_ranges = get_row_ranges(A)
	col_ranges = get_col_ranges(A)

	# for j in 1:size(A.blocks, 2) # TODO: thread outer loop?
	tforeach(1:size(A.blocks, 2)) do j # Configure scheduler?
		out_view = @view out[col_ranges[j]]
		for i in 1:size(A.blocks, 1)
			mask_view = @view mask[row_ranges[i]] # Should we use a view here? Or is it somehow inefficient to have a view into a BitVector?
			_masked_column_sum!(f, out_view, A.blocks[i,j], mask_view, f0)
		end
	end
	out
end

function _masked_row_sum!(f, out, A::Blocks, mask, f0)
	row_ranges = get_row_ranges(A)
	col_ranges = get_col_ranges(A)

	# for i in 1:size(A.blocks, 1) # TODO: thread outer loop?
	tforeach(1:size(A.blocks,1)) do i # Configure scheduler?
		out_view = @view out[row_ranges[i]]
		for j in 1:size(A.blocks, 2)
			mask_view = @view mask[col_ranges[j]] # Should we use a view here? Or is it somehow inefficient to have a view into a BitVector?
			_masked_row_sum!(f, out_view, A.blocks[i,j], mask_view, f0)
		end
	end
	out
end


_counts_sum_zero(counts::SparseMatrixCSC{T}) where T = zero(T)
_counts_sum_zero(counts::Blocks{T}) where T = zero(eltype(T))



"""
	counts_sum(f, counts, ind; dims)

See also: [`counts_fraction`](@ref)
"""
function counts_sum(f::F, counts, ind; dims::Integer, f0=f(_counts_sum_zero(counts))) where F
	@assert dims in (1,2)
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
end
