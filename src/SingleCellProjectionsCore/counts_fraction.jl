"""
	counts_fraction(counts, sub_ind, tot_ind; dims)

See also: [`counts_sum`](@ref)
"""
# function counts_fraction(counts::AbstractMatrix{<:Integer}, sub_ind, tot_ind; dims)
function counts_fraction(counts, sub_ind, tot_ind; dims)
	@assert dims in (1,2)

	# sub_mask = falses(size(counts,dims), 1)
	# tot_mask = falses(size(counts,dims), 1)
	sub_mask = fill(false, size(counts,dims), 1) # Nx1 Matrix{Bool} because that works better with Blocks
	tot_mask = fill(false, size(counts,dims), 1) # Nx1 Matrix{Bool} because that works better with Blocks
	sub_mask[sub_ind] .= true
	tot_mask[tot_ind] .= true

	@assert all(tot_mask .| .!sub_mask) # All elements in sub_mask must also be in tot_mask

	if dims == 1
		sub_count = vec(convert(Matrix,counts'sub_mask))
		tot_count = vec(convert(Matrix,counts'tot_mask))
	else#if dims == 2
		sub_count = vec(convert(Matrix,counts*sub_mask))
		tot_count = vec(convert(Matrix,counts*tot_mask))
	end

	sub_count ./ max.(1, tot_count) # avoid div by zero
end
