function find_nearest_neighbors(args...; kwargs...)
	indices, distances = SCPCore.find_nearest_neighbors(args...; kwargs...)
	CompoundResult(; indices, distances)
end

# It could be argued that this should be handled by projection, e.g. base case is X and projected case is X,action(X).
# But let's keep the separetly for now and use them as lower-level operations in other Projectables. Because it's not obvious that the above is always true.
find_nearest_neighbors_spec(X; k, kwargs...) =
	create_spec(find_nearest_neighbors, X; k, kwargs..., __version=v"0.1.0")
find_nearest_neighbors_spec(X, Y; k, kwargs...) =
	create_spec(find_nearest_neighbors, X, Y; k, kwargs..., __version=v"0.1.0")




adjacency_matrix_spec(indices; kwargs...) =
	create_spec(SCPCore.adjacency_matrix, indices; kwargs..., __version=v"0.1.0")



struct InvDistSquared
	min_dist::Float64
end
InvDistSquared() = InvDistSquared(0.0)
(x::InvDistSquared)(d::Float64) = 1.0 / max(x.min_dist, d)^2

ReproducibleJobs.copy_arg(x::InvDistSquared) = x


weighted_adjacency_matrix_spec(f, indices, dists; kwargs...) =
	create_spec(SCPCore.weighted_adjacency_matrix, f, indices, dists; kwargs..., __version=v"0.1.0")
