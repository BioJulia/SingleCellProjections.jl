function find_nearest_neighbors_impl(X, args...; k=nothing, k_fraction=nothing, kwargs...)
	k = @something k round(Int,k_fraction*size(X,2)) # In case of two data sets, k_fraction is refers to the first (base) dataset
	# @show k

	# ReproducibleJobs.add_info_item!(ReproducibleJobs.get_scheduler().progress_display, "⋅ find_nearest_neighbors_impl k=$k")

	# @time indices, distances = SCPCore.find_nearest_neighbors(X, args...; k, kwargs...)
	# CompoundResult(; indices, distances)
	# SCPCore.find_nearest_neighbors(X, args...; k, kwargs...) # returns indices only

	progress = ProgressBar(styled"{blue:  ┌─}")
	SCPCore.find_nearest_neighbors(X, args...; k, progress, kwargs...) # returns indices only
end

# It could be argued that this should be handled by projection, e.g. base case is X and projected case is X,action(X).
# But let's keep the separetly for now and use them as lower-level operations in other Projectables. Because it's not obvious that the above is always true.
function find_nearest_neighbors_spec(X, args...; k=nothing, k_fraction=nothing, kwargs...)
	@assert length(args) <= 1 "Exactly one or two matrices must be specified, got $(length(args)+1)."
	@assert (k===nothing) != (k_fraction===nothing) "Exactly one of `k` and `k_fraction` must be specified."
	k = k===nothing ? (;) : (;k)
	k_fraction = k_fraction===nothing ? (;) : (;k_fraction)
	create_spec(find_nearest_neighbors_impl, X, args...; k..., k_fraction..., kwargs..., __version=v"0.1.3")
end




adjacency_matrix_spec(indices; kwargs...) =
	create_spec(SCPCore.adjacency_matrix, indices; kwargs..., __version=v"0.1.0")


# # NB: If we change how InvDistSquared works, we must change its stable_hash too.
# struct InvDistSquared
# 	min_dist::Float64
# end
# InvDistSquared() = InvDistSquared(0.0)
# (x::InvDistSquared)(d::Float64) = 1.0 / max(x.min_dist, d)^2

# ReproducibleJobs.deduplicate_type(::Type{InvDistSquared}) = false
# ReproducibleJobs.deconstruct_weak_rec(x::InvDistSquared) = x
# ReproducibleJobs.reconstruct_weak_rec(x::InvDistSquared) = x

# function ReproducibleJobs.cache_save(io, name, x::InvDistSquared)
# 	io[name] = x # Rely on JLD2 standard handling for saving/loading
# 	nothing
# end


# # Deprecated, we should recompute distances on the fly when we need them. (Instead of caching them to disk, because it's too much data.)
# weighted_adjacency_matrix_spec(f, indices, dists; kwargs...) =
# 	create_spec(SCPCore.weighted_adjacency_matrix, f, indices, dists; kwargs..., __version=v"0.1.0")


# NB: If we change how InvMax works, we must change its stable_hash too.
struct InvMax
	min::Float64
end
# InvMax() = InvMax(0.0)
(x::InvMax)(d::Float64) = 1.0 / max(x.min, d)

ReproducibleJobs.deduplicate_type(::Type{InvMax}) = false
ReproducibleJobs.deconstruct_weak_rec(x::InvMax) = x
ReproducibleJobs.reconstruct_weak_rec(x::InvMax) = x

function ReproducibleJobs.cache_save(io, name, x::InvMax)
	io[name] = x # Rely on JLD2 standard handling for saving/loading
	nothing
end

