annotate(::Mat, data; kwargs...) = SCP.get_matrix(data)
function annotate(f::Union{Var,Obs}, data; kwargs...)
	s = get_job(f, data)
	df = get(kwargs, f isa Var ? :var : :obs, nothing)
	df === nothing && return s
	SCP.table_leftjoin(s, df)
end


add_var_column(f::Union{Mat,Obs}, data, name, column) = get_job(f, data)
add_var_column(::Var, data, name, column) = SCP.add_column(SCP.get_var(data), name, column)

add_obs_column(f::Union{Mat,Var}, data, name, column) = get_job(f, data)
add_obs_column(::Obs, data, name, column) = SCP.add_column(SCP.get_obs(data), name, column)





counts_fraction_impl_job(counts, sub_ind, tot_ind; dims) =
	create_job(SCPCore.counts_fraction, counts, sub_ind, tot_ind; dims, __version=v"1.0.0")

counts_sum_impl_job(f, counts, ind; dims) =
	create_job(SCPCore.counts_sum, f, counts, ind; dims, __version=v"1.0.0")

# Block-aware reduction, computing (and caching) `counts_sum` per block for cross-dataset cache reuse.
# dims=1 (per-obs result): the row (var) mask `ind` is identical for every column block; combine the
# disjoint per-obs results with `vcat`. dims=2 (per-var result): `ind` selects obs (columns), so it is
# split per block (block-local) via `ind_to_blocked_ind` and the partial per-var sums are combined
# element-wise with `vsum`. Falls back to a single cached job when the spec is not block-structured.
function counts_sum_blocked(::Preprocessing, f, X, ind; dims)
	@assert dims in (1,2)
	if dims == 1
		hblock_map(X; wrap=(a,_)->vcat_job(a)) do x
			cached(counts_sum_impl_job(f, x, ind; dims))
		end
	elseif is_hblock(X)
		blocks = X.args[1]
		ranges = _get_kwarg(X, :ranges)
		block_ind, _ = SCPCore.ind_to_blocked_ind(ind, ranges) # per-block, block-local obs selection
		vsum_job([cached(counts_sum_impl_job(f, b, I; dims)) for (b,I) in zip(blocks, block_ind)])
	else
		cached(counts_sum_impl_job(f, X, ind; dims))
	end
end
# For dims=2, `ind` must be a concrete value during preprocessing (so it can be split per block), hence
# `fetched`; for dims=1 it stays a lazy job argument.
counts_sum_blocked_job(f, X, ind; dims) =
	create_job(Preprocess{false}(counts_sum_blocked), f, X, dims==1 ? ind : fetched(ind); dims)




var_counts_fraction(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
var_counts_fraction(::Var, counts, args...; kwargs...) = SCP.get_var(counts)
function var_counts_fraction(::Obs, counts, col, sub_filter, tot_filter; project_ids)
	var_job = SCP.get_var(counts)
	sub_ind = prefetched(create_find_matching_ind_job(sub_filter, var_job; project_ids))
	tot_ind = prefetched(create_find_matching_ind_job(tot_filter, var_job; project_ids))
	values_job = cached(counts_fraction_impl_job(SCP.get_matrix(counts), sub_ind, tot_ind; dims=1))
	SCP.add_column(SCP.get_obs(counts), col, values_job)
end


var_counts_sum(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
var_counts_sum(::Var, counts, args...; kwargs...) = SCP.get_var(counts)
function var_counts_sum(::Obs, counts, col, filter; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_job(filter, SCP.get_var(counts); project_ids))
	values_job = counts_sum_blocked_job(f, SCP.get_matrix(counts), ind; dims=1)
	SCP.add_column(SCP.get_obs(counts), col, values_job)
end



# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_fraction(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
function obs_counts_fraction(::Var, counts, col, sub_filter, tot_filter; project_ids)
	obs_job = SCP.get_obs(counts)
	sub_ind = prefetched(create_find_matching_ind_job(sub_filter, obs_job; project_ids))
	tot_ind = prefetched(create_find_matching_ind_job(tot_filter, obs_job; project_ids))

	values_job = cached(counts_fraction_impl_job(SCP.get_matrix(counts), sub_ind, tot_ind; dims=2))
	SCP.add_column(SCP.get_var(counts), col, values_job)
end
obs_counts_fraction(::Obs, counts, args...; kwargs...) = SCP.get_obs(counts)


# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_sum(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
function obs_counts_sum(::Var, counts, col, filter; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_job(filter, SCP.get_obs(counts); project_ids))
	values_job = counts_sum_blocked_job(f, SCP.get_matrix(counts), ind; dims=2)
	SCP.add_column(SCP.get_var(counts), col, values_job)
end
obs_counts_sum(::Obs, counts, args...; kwargs...) = SCP.get_obs(counts)
