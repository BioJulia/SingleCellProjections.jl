function local_reachability_density(dists, kdists)
	N1,N2 = size(dists)
	@assert length(kdists) == N1
	lrd = zeros(N2)

	R = rowvals(dists)
	V = nonzeros(dists)
	for j=1:N2
		rd_sum = 0.0
		for k in nzrange(dists,j)
			i = R[k]
			d = V[k]
			rd = max(kdists[i], d) # reachability distance from point j to point i
			rd_sum += rd
		end
		lrd[j] = length(nzrange(dists,j))/rd_sum
	end

	lrd
end


# adj can be either the distance matrix or the adjacency matrix - since only the sparsity pattern is considered
function _local_outlier_factor(adj::AbstractSparseMatrix, lrdX::AbstractVector, lrdY::AbstractVector=lrdX)
	N1,N2 = size(adj)
	@assert length(lrdX) == N1
	@assert length(lrdY) == N2

	lof = zeros(N2)

	R = rowvals(adj)
	for j=1:N2
		lrdX_sum = 0.0
		for k in nzrange(adj,j)
			i = R[k]
			lrdX_sum += lrdX[i]
		end
		lof[j] = lrdX_sum/(length(nzrange(adj,j)) * lrdY[j])
	end

	lof
end





"""
	local_outlier_factor!(data::DataMatrix, full::DataMatrix; k=10, col="LOF")

Compute the Local Outlier Factor (LOF) for each observation in `data`, and add as column to
`data.obs` with the name `col`.

When working with projected `DataMatrices`, use [`local_outlier_factor_projection!`](@ref)
instead.

NB: This function might be very slow for high values of `k`.

First, the `k` nearest neighbors are found for each observation in `data`.
Then, the Local Outlier Factor is computed by considering the distance between the
neighbors, but this time in the `full` DataMatrix. Thus `full` must have the same
observations as are present in `data`.

A LOF value smaller than or close to one is means that the observation is an inlier, but a
LOF value much larger than one means that the observation is an inlier.

By specifiying `full=data`, this is coincides with the standard definition for Local Outlier
Factor.
However, it is perhaps more useful to find neighbors in a dimension reduced space (after
e.g. `svd` (PCA) or `umap`), but then compute distances in the high dimensional space
(typically after normalization).
This way, an observation is concidered an outlier if the reduction to a lower dimensional
space didn't properly represent the neighborhood of the observation.

!!! note
	The interface is not yet fully decided and subject to change.

# Examples

Compute the Local Outlier Factor, with nearest neighbors based only on `reduced`, but later
using distances in `full` for the actual LOF computation.

```julia
julia> reduced = svd(normalized; nsv=10)

julia> local_outlier_factor!(reduced, normalized; k=10)
```

See also: [`local_outlier_factor_projection!`](@ref)
"""
function local_outlier_factor!(data::DataMatrix, full::DataMatrix; k=10, col="LOF")
	adj = knn_adjacency_matrix(data; k, make_symmetric=false)
	dists = adjacency_distances(adj, full)
	kdists = vec(maximum(dists.matrix; dims=1))
	lrd = local_reachability_density(dists.matrix, kdists);
	data.obs[:,col] = _local_outlier_factor(dists.matrix, lrd)
	data
end

"""
	local_outlier_factor_projection!(data::DataMatrix, full::DataMatrix, base::DataMatrix, base_full::DataMatrix; k=10, col="LOF_projection")

Compute the Local Outlier Factor (LOF) for each observation in `data`, and add as column to
`data.obs` with the name `col`.

Use `local_outlier_factor_projection!` if you are working with projected data, and
[`local_outlier_factor!`](@ref) otherwise.

Parameters:

* `data` - A `DataMatrix` for which we compute LOF for each observation. Expected to be a `DataMatrix` projected onto `base`, so that the `data` and `base` use the same coordinate system.
* `full` - A `DataMatrix` with the same observations as `data`, used to compute distances in the LOF computation. Expected to be a `DataMatrix` projected onto `base_full`, so that the `full` and `base_full` use the same coordinate system.
* `base` - The base `DataMatrix`.
* `base_full` - The base `DataMatrix`.
* `k` - The number of nearest neighbors to use. NB: This function might be very slow for high values of `k`.

First, for each observation in `data`, the `k` nearest neighbors in `base` are found.
Then, the distance to each neighbor is computed using `full` and `base_full`.
Thus `full` must have the same observations as are present in `data`, and `base_full` must
have the same observations as `base`.

A LOF value smaller than or close to one is means that the observation is an inlier, but a
LOF value much larger than one means that the observation is an inlier.

By specifiying `full=data` and `base_full=base`, this is coincides with the standard
definition for Local Outlier Factor.
However, it is perhaps more useful to find neighbors in a dimension reduced space (after
e.g. `svd` (PCA) or `umap`), but then compute distances in the high dimensional space
(typically after normalization).
This way, an observation is concidered an outlier if the reduction to a lower dimensional
space didn't properly represent the neighborhood of the observation.

!!! note
	The interface is not yet fully decided and subject to change.


# Examples

Compute the Local Outlier Factor (LOF) for each observation in a data set `reduced`, which
has been projected onto `base_reduced`.

The nearest neighbors are computed between observations in `reduced` and `base_reduced`, but
the distances in the actual LOF computation are between the same observations in `normalized` and `base_normalized`.

```julia
julia> base_reduced = svd(base_normalized; nsv=10)

julia> normalized = project(counts, base_normalized);

julia> reduced = project(normalized, base_reduced);

julia> local_outlier_factor!(reduced, normalized, base_reduced, base_normalized; k=10)
```

See also: [`local_outlier_factor!`](@ref)
"""
function local_outlier_factor_projection!(data::DataMatrix, full::DataMatrix,
                                          base::DataMatrix, base_full::DataMatrix;
                                          k=10,
                                          col="LOF_projection")
	adj_X = knn_adjacency_matrix(base; k, make_symmetric=false)
	adj_XY = knn_adjacency_matrix(base, data; k)

	dists_X = adjacency_distances(adj_X, base_full)
	dists_XY = adjacency_distances(adj_XY, base_full, full)

	kdists = vec(maximum(dists_X.matrix; dims=1))
	
	lrd_X = local_reachability_density(dists_X.matrix, kdists);
	lrd_XY = local_reachability_density(dists_XY.matrix, kdists);

	data.obs[:,col] = _local_outlier_factor(dists_XY.matrix, lrd_X, lrd_XY)
	data
end