# """
# 	adjacency_distances(adj, X, Y=X)

# For each structural non-zero in `adj`, compute the Euclidean distance between the point in
# the DataMatrix `Y` and the point in the DataMatrix `X`.

# Can be useful when `adj` is created using e.g. a lower dimensional representation and we
# want to know the distances in the original, high dimensional space.

# At the moment all points in `Y` are required to have the same number of neighbors in `X`,
# for computation reasons.
# """
# function adjacency_distances(adj::DataMatrix, X::DataMatrix, Y::DataMatrix=X)
# 	table_cols_equal(adj.var, X.obs; cols=names(X.obs,1)) || error("Adjacency matrix and DataMatrix have different obs.")
# 	table_cols_equal(adj.obs, Y.obs; cols=names(Y.obs,1)) || error("Adjacency matrix and DataMatrix have different obs.")
# 	D = _adjacency_distances(adj.matrix, X, Y)
# 	DataMatrix(D, copy(adj.var), copy(adj.obs))
# end


# function _adjacency_distances(adj, X::DataMatrix, Y::DataMatrix=X)
# 	N1,N2 = size(adj)
# 	@assert size(X,2)==N1
# 	@assert size(Y,2)==N2

# 	nnbors = vec(sum(!iszero, adj; dims=1))
# 	@assert all(isequal(nnbors[1]), nnbors) "All points in Y must have the same number of neighbors in X"
# 	k = nnbors[1]
# 	@assert k>0

# 	DX2 = compute(DiagGram(X.matrix))
# 	DY2 = compute(DiagGram(Y.matrix))

# 	I,J,_ = findnz(adj)
# 	dists = zeros(length(I)) # output

# 	for i in 1:k
# 		Is = I[i:k:end]
# 		# Js = J[i:k:end] # guaranteed to be 1:N2

# 		# Xs = X[:,Is] # Doesn't work since DataMatrix doesn't allow duplicate IDs
# 		# Temporary workaround - TODO: fix proper interface?
# 		Xs = DataMatrix(_subsetmatrix(X.matrix,:,Is), X.var, DataFrame(id=1:length(Is)))


# 		# Ys = Y[:,Js] # guaranteed to be equal to Y

# 		DXYs = compute(Diag(matrixproduct(Xs.matrix',Y.matrix)))

# 		DX2s = DX2[Is]

# 		dists[i:k:end] .= sqrt.(max.(0.0, DX2s .+ DY2 .- 2DXYs))
# 	end

# 	sparse(I,J,dists,N1,N2)
# end

# TODO: Move these to nearest neighbor file?
col_sum_squared(X) = compute(DiagGram(X))
row_sum_squared(X) = compute(DiagGram(X'))

# TODO: Move this to nearest neighbor file?
function neighbor_distances(indices, X, DX2, Y=X, DY2=DX2; progress=nothing)
	k = size(indices,1)
	NX = size(X,2)
	NY = size(Y,2)

	@assert length(DX2) == NX
	@assert length(DY2) == NY
	@assert size(indices,2) == NY


	# TODO: Make some utility function to match the blocking of a matrix expression instead???
	@assert X isa MatrixSum
	@assert X.terms[1] isa MatrixRef
	@assert X.terms[1].matrix isa Blocks
	@assert Y isa MatrixSum
	@assert Y.terms[1] isa MatrixRef
	@assert Y.terms[1].matrix isa Blocks
	X_col_ranges = get_col_ranges(X.terms[1].matrix)
	Y_col_ranges = get_col_ranges(Y.terms[1].matrix)

	# progress = verbose ? Progress(k; desc="Computing distances to neighbors: ") : nothing
	isnothing(progress) || progress(k) # initialize

	dists = zeros(k, NY) # TODO: Use Float32?
	for i in 1:k
		ind = indices[i,:]

		S = sparse(ind, 1:NY, true, NX, NY)
		# blockify to match the obs block sizes of both X and Y. TODO: Construct directly without first materializing full sparse matrix?
		S = blockify(S; row_ranges=X_col_ranges, col_ranges=Y_col_ranges)

		Xs = matrixproduct(X, :S=>S) # X reordered so column j of Xs contains the i'th neighbor of column j in Y

		DXYs = compute(Diag(matrixproduct(Xs',Y)))
		DX2s = DX2[ind]
		dists[i,:] .= sqrt.(max.(0.0, DX2s .+ DY2 .- 2DXYs))
		isnothing(progress) || progress() # step
	end
	dists
end



function local_reachability_density(indices, dists, kdists)
	n_neighbors = size(indices,1)
	# NX = size(kdists,2)
	NY = size(indices,2)
	@assert size(dists,1) == n_neighbors
	@assert size(dists,2) == NY
	@assert size(kdists,1) == 1

	lrd = zeros(NY)

	for j in 1:NY # TODO: Parallelize?
		rd_sum = 0.0
		for k in 1:n_neighbors
			i = indices[k,j]
			d = dists[k,j]
			rd = max(kdists[i], d) # reachability distance from point j to point i
			rd_sum += rd
		end
		lrd[j] = n_neighbors/rd_sum
	end
	lrd
end



function local_outlier_factor(indices, lrdX, lrdY=lrdX)
	n_neighbors = size(indices,1)
	NX = length(lrdX)
	NY = length(lrdY)
	@assert size(indices,2) == NY
	
	lof = zeros(NY)

	for j=1:NY
		lrdX_sum = 0.0
		for k in 1:n_neighbors
			i = indices[k,j]
			lrdX_sum += lrdX[i]
		end
		lof[j] = lrdX_sum/(n_neighbors * lrdY[j])
	end

	lof
end







# function local_reachability_density(dists, kdists)
# 	N1,N2 = size(dists)
# 	@assert length(kdists) == N1
# 	lrd = zeros(N2)

# 	R = rowvals(dists)
# 	V = nonzeros(dists)
# 	for j=1:N2
# 		rd_sum = 0.0
# 		for k in nzrange(dists,j)
# 			i = R[k]
# 			d = V[k]
# 			rd = max(kdists[i], d) # reachability distance from point j to point i
# 			rd_sum += rd
# 		end
# 		lrd[j] = length(nzrange(dists,j))/rd_sum
# 	end

# 	lrd
# end


# # adj can be either the distance matrix or the adjacency matrix - since only the sparsity pattern is considered
# function _local_outlier_factor(adj::AbstractSparseMatrix, lrdX::AbstractVector, lrdY::AbstractVector=lrdX)
# 	N1,N2 = size(adj)
# 	@assert length(lrdX) == N1
# 	@assert length(lrdY) == N2

# 	lof = zeros(N2)

# 	R = rowvals(adj)
# 	for j=1:N2
# 		lrdX_sum = 0.0
# 		for k in nzrange(adj,j)
# 			i = R[k]
# 			lrdX_sum += lrdX[i]
# 		end
# 		lof[j] = lrdX_sum/(length(nzrange(adj,j)) * lrdY[j])
# 	end

# 	lof
# end


# function _local_outlier_factor(data::DataMatrix, full::DataMatrix; k)
# 	adj = knn_adjacency_matrix(data; k, make_symmetric=false)
# 	dists = adjacency_distances(adj, full)
# 	kdists = vec(maximum(dists.matrix; dims=1))
# 	lrd = local_reachability_density(dists.matrix, kdists);
# 	_local_outlier_factor(dists.matrix, lrd)
# end





# """
# 	local_outlier_factor!(data::DataMatrix, full::DataMatrix; k=10, col="LOF")

# Compute the Local Outlier Factor (LOF) for each observation in `data`, and add as column to
# `data.obs` with the name `col`.

# When working with projected `DataMatrices`, use [`local_outlier_factor_projection!`](@ref)
# instead.

# NB: This function might be very slow for high values of `k`.

# First, the `k` nearest neighbors are found for each observation in `data`.
# Then, the Local Outlier Factor is computed by considering the distance between the
# neighbors, but this time in the `full` DataMatrix. Thus `full` must have the same
# observations as are present in `data`.

# A LOF value smaller than or close to one is means that the observation is an inlier, but a
# LOF value much larger than one means that the observation is an inlier.

# By specifiying `full=data`, this is coincides with the standard definition for Local Outlier
# Factor.
# However, it is perhaps more useful to find neighbors in a dimension reduced space (after
# e.g. `svd` (PCA) or `umap`), but then compute distances in the high dimensional space
# (typically after normalization).
# This way, an observation is concidered an outlier if the reduction to a lower dimensional
# space didn't properly represent the neighborhood of the observation.

# !!! note
# 	The interface is not yet fully decided and subject to change.

# # Examples

# Compute the Local Outlier Factor, with nearest neighbors based only on `reduced`, but later
# using distances in `full` for the actual LOF computation.

# ```julia
# julia> reduced = svd(normalized; nsv=10)

# julia> local_outlier_factor!(reduced, normalized; k=10)
# ```

# See also: [`local_outlier_factor`](@ref), [`local_outlier_factor_table`](@ref), [`local_outlier_factor_projection!`](@ref)
# """
# function local_outlier_factor!(data::DataMatrix, full::DataMatrix; k=10, col="LOF")
# 	data.obs[:,col] = _local_outlier_factor(data, full; k)
# 	data
# end


# """
# 	local_outlier_factor(data::DataMatrix, full::DataMatrix; k=10, col="LOF", matrix=:keep, var=:copy)

# See `local_outlier_factor!` for documentation. This version does not modify `data` in place.

# See also: [`local_outlier_factor!`](@ref), [`local_outlier_factor_table`](@ref), [`local_outlier_factor_projection`](@ref)
# """
# function local_outlier_factor(data::DataMatrix, full::DataMatrix; k=10, col="LOF", matrix=:keep, var=:copy, kwargs...)
# 	obs = copy(data.obs)
# 	obs[:,col] = _local_outlier_factor(data, full; k)
# 	X = matrix == :keep ? data.matrix : copy(data.matrix)
# 	var = _update_annot(data.var, var, size(matrix,1))
# 	DataMatrix(X, var, obs, data.models; kwargs...)
# end


# """
# 	local_outlier_factor_table(data::DataMatrix, full::DataMatrix; k=10, col="LOF")

# See `local_outlier_factor!` for documentation. This returns a DataFrame with observation IDs and a column `col` with LOF values.

# See also: [`local_outlier_factor!`](@ref), [`local_outlier_factor`](@ref), [`local_outlier_factor_projection_table`](@ref)
# """
# function local_outlier_factor_table(data::DataMatrix, full::DataMatrix; k=10, col="LOF")
# 	df = select(data.obs, 1)
# 	df[:,col] = _local_outlier_factor(data, full; k)
# 	df
# end





# function _local_outlier_factor_projection(data::DataMatrix, full::DataMatrix,
#                                           base::DataMatrix, base_full::DataMatrix;
#                                           k=10)
# 	adj_X = knn_adjacency_matrix(base; k, make_symmetric=false)
# 	adj_XY = knn_adjacency_matrix(base, data; k)

# 	dists_X = adjacency_distances(adj_X, base_full)
# 	dists_XY = adjacency_distances(adj_XY, base_full, full)

# 	kdists = vec(maximum(dists_X.matrix; dims=1))

# 	lrd_X = local_reachability_density(dists_X.matrix, kdists);
# 	lrd_XY = local_reachability_density(dists_XY.matrix, kdists);

# 	_local_outlier_factor(dists_XY.matrix, lrd_X, lrd_XY)
# end


# """
# 	local_outlier_factor_projection!(data::DataMatrix, full::DataMatrix, base::DataMatrix, base_full::DataMatrix; k=10, col="LOF_projection")

# Compute the Local Outlier Factor (LOF) for each observation in `data`, and add as column to
# `data.obs` with the name `col`.

# Use `local_outlier_factor_projection!` if you are working with projected data, and
# [`local_outlier_factor!`](@ref) otherwise.

# Parameters:

# * `data` - A `DataMatrix` for which we compute LOF for each observation. Expected to be a `DataMatrix` projected onto `base`, so that the `data` and `base` use the same coordinate system.
# * `full` - A `DataMatrix` with the same observations as `data`, used to compute distances in the LOF computation. Expected to be a `DataMatrix` projected onto `base_full`, so that the `full` and `base_full` use the same coordinate system.
# * `base` - The base `DataMatrix`.
# * `base_full` - The base `DataMatrix`.
# * `k` - The number of nearest neighbors to use. NB: This function might be very slow for high values of `k`.

# First, for each observation in `data`, the `k` nearest neighbors in `base` are found.
# Then, the distance to each neighbor is computed using `full` and `base_full`.
# Thus `full` must have the same observations as are present in `data`, and `base_full` must
# have the same observations as `base`.

# A LOF value smaller than or close to one is means that the observation is an inlier, but a
# LOF value much larger than one means that the observation is an inlier.

# By specifiying `full=data` and `base_full=base`, this is coincides with the standard
# definition for Local Outlier Factor.
# However, it is perhaps more useful to find neighbors in a dimension reduced space (after
# e.g. `svd` (PCA) or `umap`), but then compute distances in the high dimensional space
# (typically after normalization).
# This way, an observation is concidered an outlier if the reduction to a lower dimensional
# space didn't properly represent the neighborhood of the observation.

# !!! note
# 	The interface is not yet fully decided and subject to change.


# # Examples

# Compute the Local Outlier Factor (LOF) for each observation in a data set `reduced`, which
# has been projected onto `base_reduced`.

# The nearest neighbors are computed between observations in `reduced` and `base_reduced`, but
# the distances in the actual LOF computation are between the same observations in `normalized` and `base_normalized`.

# ```julia
# julia> base_reduced = svd(base_normalized; nsv=10)

# julia> normalized = project(counts, base_normalized);

# julia> reduced = project(normalized, base_reduced);

# julia> local_outlier_factor!(reduced, normalized, base_reduced, base_normalized; k=10)
# ```

# See also: [`local_outlier_factor_projection`](@ref), [`local_outlier_factor_projection_table`](@ref), [`local_outlier_factor!`](@ref)
# """
# function local_outlier_factor_projection!(data::DataMatrix, full::DataMatrix,
#                                           base::DataMatrix, base_full::DataMatrix;
#                                           k=10,
#                                           col="LOF_projection")
# 	data.obs[:,col] = _local_outlier_factor_projection(data, full, base, base_full; k)
# 	data
# end




# """
# 	local_outlier_factor_projection(data::DataMatrix, full::DataMatrix, base::DataMatrix, base_full::DataMatrix; k=10, col="LOF_projection", matrix=:keep, var=:copy)

# See `local_outlier_factor_projection!` for documentation. This version does not modify `data` in place.

# See also: [`local_outlier_factor_projection!`](@ref), [`local_outlier_factor_projection_table`](@ref), [`local_outlier_factor`](@ref)
# """
# function local_outlier_factor_projection(data::DataMatrix, full::DataMatrix,
#                                          base::DataMatrix, base_full::DataMatrix;
#                                          k=10,
#                                          col="LOF_projection",
#                                          matrix = :keep,
#                                          var = :copy,
#                                          kwargs...)
# 	obs = copy(data.obs)
# 	obs[:,col] = _local_outlier_factor_projection(data, full, base, base_full; k)
# 	X = matrix == :keep ? data.matrix : copy(data.matrix)
# 	var = _update_annot(data.var, var, size(matrix,1))
# 	DataMatrix(X, var, obs, data.models; kwargs...)
# end


# """
# 	local_outlier_factor_projection_table(data::DataMatrix, full::DataMatrix, base::DataMatrix, base_full::DataMatrix; k=10, col="LOF_projection", matrix=:keep, var=:copy)

# See `local_outlier_factor_projection!` for documentation. This returns a DataFrame with observation IDs and a column `col` with LOF values for the projection.

# See also: [`local_outlier_factor_projection!`](@ref), [`local_outlier_factor_projection`](@ref), [`local_outlier_factor_table`](@ref)
# """
# function local_outlier_factor_projection_table(data::DataMatrix, full::DataMatrix,
#                                                base::DataMatrix, base_full::DataMatrix;
#                                                k=10,
#                                                col="LOF_projection")
# 	df = select(data.obs, 1)
# 	df[:,col] = _local_outlier_factor_projection(data, full, base, base_full; k)
# 	df
# end
