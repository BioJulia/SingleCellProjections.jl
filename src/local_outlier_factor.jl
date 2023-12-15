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
	local_outlier_factor!(data::DataMatrix, full::DataMatrix)

Local Outlier Factor computation.

The interface is not yet fully decided and subject to change.
"""
function local_outlier_factor!(data::DataMatrix, full::DataMatrix; k, col="LOF")
	adj = knn_adjacency_matrix(data; k, make_symmetric=false)
	dists = adjacency_distances(adj, full)
	kdists = vec(maximum(dists.matrix; dims=1))
	lrd = local_reachability_density(dists.matrix, kdists);
	data.obs[:,col] = _local_outlier_factor(dists.matrix, lrd)
	data
end

"""
	local_outlier_factor_projection!(data::DataMatrix, full::DataMatrix, base::DataMatrix, base_full::DataMatrix)

Local Outlier Factor computation, considering only neighbors in the base data set.

The interface is not yet fully decided and subject to change.
"""
function local_outlier_factor_projection!(data::DataMatrix, full::DataMatrix,
                                          base::DataMatrix, base_full::DataMatrix;
                                          k, col="LOF_projection")
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