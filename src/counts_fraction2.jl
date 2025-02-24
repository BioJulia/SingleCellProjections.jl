function var_counts_fraction2(counts::AbstractMatrix{<:Integer}, sub_ind, tot_ind)
	sub_mask = falses(size(counts,1))
	tot_mask = falses(size(counts,1))
	sub_mask[sub_ind] .= true
	tot_mask[tot_ind] .= true

	@assert all(tot_mask .| .!sub_mask) # All elements in sub_mask must also be in tot_mask

	sub_count = vec(sub_mask'counts)
	tot_count = vec(tot_mask'counts)

	sub_count ./ max.(1, tot_count) # avoid div by zero
end

