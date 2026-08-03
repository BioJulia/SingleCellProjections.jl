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





counts_sum_impl_job(f, counts, ind; dims) =
	create_job(SCPCore.counts_sum, f, counts, ind; dims, __version=v"1.0.0")

# Block-aware reduction, computing (and caching) `counts_sum` per block for cross-dataset cache reuse.
# dims=1 (per-obs result): the row (var) mask `ind` is identical for every column block; combine the
# disjoint per-obs results with `vcat`. dims=2 (per-var result): `ind` selects obs (columns), so it is
# split per block (block-local) via `ind_to_blocked_ind` and the partial per-var sums are combined
# element-wise with `sum` (apply_job). Falls back to a single cached job when the spec is not block-structured.
# The per-block results are cached (in the wrap, so a recurring sample dedups across datasets); the
# combine (vcat/sum) is left uncached, and the non-block fallback returns the *uncached* leaf. This lets
# the caller decide whether to cache the combined result (`cached(counts_sum_job(...))`) without nesting
# `cached` in the non-block case (see var_counts_sum etc.).
function counts_sum(::Preprocessing, f, X, ind; dims)
	@assert dims in (1,2)
	if dims == 1
		# `ind` is the row (var) mask, identical for every column block; combine disjoint per-obs results.
		hblock_map(X; wrap=(a,_)->vcat_job(cached.(a))) do x
			counts_sum_impl_job(f, x, ind; dims)
		end
	else
		# `ind` selects obs (columns): split it per block (block-local) and sum the partial per-var results.
		block_ind = is_hblock(X) ? first(SCPCore.ind_to_blocked_ind(ind, _get_kwarg(X, :ranges))) : [ind]
		hblock_map(X, block_ind; wrap=(a,_)->apply_job(sum, cached.(a))) do x, I
			counts_sum_impl_job(f, x, I; dims)
		end
	end
end
# For dims=2, `ind` must be a concrete value during preprocessing (so it can be split per block), hence
# `fetched`; for dims=1 it stays a lazy job argument.
counts_sum_job(f, X, ind; dims) =
	create_job(Preprocess{false}(counts_sum), f, X, dims==1 ? ind : fetched(ind); dims)


# Combine per-obs (or per-var) sub/tot count vectors into a fraction, flooring the denominator at 1 to
# avoid division by zero.
counts_fraction_combine(sub, tot) = sub ./ max.(1, tot)
counts_fraction_combine_job(sub, tot) = create_job(counts_fraction_combine, sub, tot; __version=v"1.0.0")


var_counts_fraction(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
var_counts_fraction(::Var, counts, args...; kwargs...) = SCP.get_var(counts)
function var_counts_fraction(::Obs, counts, col, sub_filter, tot_filter; project_ids)
	var_job = SCP.get_var(counts)
	sub_ind = prefetched(create_find_matching_ind_job(sub_filter, var_job; project_ids))
	tot_ind = prefetched(create_find_matching_ind_job(tot_filter, var_job; project_ids))

	# TODO: Consider enforcing that sub_ind are a subset of tot_ind.

	# fraction = (sum over sub vars) / (sum over tot vars), reusing the block-aware counts_sum for both
	# so the per-block sums cache/dedup (the denominator in particular is often shared across calls).
	X = SCP.get_matrix(counts)
	sub = counts_sum_job(identity, X, sub_ind; dims=1)
	tot = counts_sum_job(identity, X, tot_ind; dims=1)
	SCP.add_column(SCP.get_obs(counts), col, cached(counts_fraction_combine_job(sub, tot)))
end


var_counts_sum(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
var_counts_sum(::Var, counts, args...; kwargs...) = SCP.get_var(counts)
function var_counts_sum(::Obs, counts, col, filter; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_job(filter, SCP.get_var(counts); project_ids))
	values_job = cached(counts_sum_job(f, SCP.get_matrix(counts), ind; dims=1))
	SCP.add_column(SCP.get_obs(counts), col, values_job)
end



# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_fraction(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
function obs_counts_fraction(::Var, counts, col, sub_filter, tot_filter; project_ids)
	obs_job = SCP.get_obs(counts)
	sub_ind = prefetched(create_find_matching_ind_job(sub_filter, obs_job; project_ids))
	tot_ind = prefetched(create_find_matching_ind_job(tot_filter, obs_job; project_ids))

	# TODO: Consider enforcing that sub_ind are a subset of tot_ind.

	# fraction = (sum over sub obs) / (sum over tot obs), reusing the block-aware counts_sum for both.
	X = SCP.get_matrix(counts)
	sub = counts_sum_job(identity, X, sub_ind; dims=2)
	tot = counts_sum_job(identity, X, tot_ind; dims=2)
	SCP.add_column(SCP.get_var(counts), col, cached(counts_fraction_combine_job(sub, tot)))
end
obs_counts_fraction(::Obs, counts, args...; kwargs...) = SCP.get_obs(counts)


# TODO: Can we get better code reuse with the `var` functions above?
obs_counts_sum(::Mat, counts, args...; kwargs...) = SCP.get_matrix(counts)
function obs_counts_sum(::Var, counts, col, filter; project_ids, f=identity)
	ind = prefetched(create_find_matching_ind_job(filter, SCP.get_obs(counts); project_ids))
	values_job = cached(counts_sum_job(f, SCP.get_matrix(counts), ind; dims=2))
	SCP.add_column(SCP.get_var(counts), col, values_job)
end
obs_counts_sum(::Obs, counts, args...; kwargs...) = SCP.get_obs(counts)
