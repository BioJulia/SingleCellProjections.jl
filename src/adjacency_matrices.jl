function knn_adjacency_matrix(X; k, make_symmetric)
	N = size(X,2)
	k = min(k,N-1)
	tree = KDTree(X)
	indices,_ = knn(tree, X, k+1)

	I = zeros(Int, k*N)
	J = zeros(Int, k*N)
	destInd = 1
	for (j,ind) in enumerate(indices)
		skip = 0
		for a in 1:k
			ind[a]==j && (skip=1)
			I[destInd] = ind[a+skip]
			J[destInd] = j
			destInd += 1
		end
	end
	adj = sparse(I,J,true,N,N)
	if make_symmetric
		adj = adj.|adj'
	end
	adj
end

function knn_adjacency_matrix(X, Y; k)
	# TODO: Avoid some code duplication by merging this with other functions? (knn_adjacency_matrix and embed_points)
	N1 = size(X,2)
	N2 = size(Y,2)
	@assert size(X,1) == size(Y,1)
	k = min(N1,k) # cannot have more neighbors than points!

	tree = KDTree(X) # Choose BallTree if dimension is higher?
	indices,dists = knn(tree, Y, k) # TODO: use threads?

	# N1xN2 sparse adjacency matrix between two matrices
	I = reduce(vcat,indices)
	J = repeat(1:N2; inner=k)

	sparse(I,J,true,N1,N2)
end


function knn_adjacency_matrix(data::DataMatrix; kwargs...)
	adj = knn_adjacency_matrix(obs_coordinates(data.matrix); kwargs...)
	obs = copy(data.obs)
	DataMatrix(adj, obs, obs; var_id_cols=data.obs_id_cols, data.obs_id_cols)
end

function knn_adjacency_matrix(X::DataMatrix, Y::DataMatrix; kwargs...)
	adj = knn_adjacency_matrix(obs_coordinates(X.matrix), obs_coordinates(Y.matrix); kwargs...)
	DataMatrix(adj, copy(X.obs), copy(Y.obs); var_id_cols=X.obs_id_cols, Y.obs_id_cols)
end





"""
	adjacency_distances(adj, X, Y=X)

For each structural non-zero in `adj`, compute the Euclidean distance between the point in
the DataMatrix `Y` and the point in the DataMatrix `X`.

Can be useful when `adj` is created using e.g. a lower-dimensional representation and we
want to know the distances in the original, high-dimensional space.

At the moment all points in `Y` are required to have the same number of neighbors in `X`,
for computation reasons.
"""
function adjacency_distances(adj::DataMatrix, X::DataMatrix, Y::DataMatrix=X)
	table_cols_equal(adj.var, X.obs; cols=X.obs_id_cols) || error("Adjacency matrix and DataMatrix have different obs.")
	table_cols_equal(adj.obs, Y.obs; cols=Y.obs_id_cols) || error("Adjacency matrix and DataMatrix have different obs.")
	D = _adjacency_distances(adj.matrix, X, Y)
	DataMatrix(D, copy(adj.var), copy(adj.obs); adj.var_id_cols, adj.obs_id_cols)
end


function _adjacency_distances(adj, X::DataMatrix, Y::DataMatrix=X)
	N1,N2 = size(adj)
	@assert size(X,2)==N1
	@assert size(Y,2)==N2

	nnbors = vec(sum(!iszero, adj; dims=1))
	@assert all(isequal(nnbors[1]), nnbors) "All points in Y must have the same number of neighbors in X"
	k = nnbors[1]
	@assert k>0

	DX2 = compute(DiagGram(X.matrix))
	DY2 = compute(DiagGram(Y.matrix))

	I,J,_ = findnz(adj)
	dists = zeros(length(I)) # output

	for i in 1:k
		Is = I[i:k:end]
		# Js = J[i:k:end] # guaranteed to be 1:N2

		# Xs = X[:,Is] # Doesn't work since DataMatrix doesn't allow duplicate IDs
		# Temporary workaround - TODO: fix proper interface?
		Xs = DataMatrix(_subsetmatrix(X.matrix,:,Is), X.var, DataFrame(id=1:length(Is)); var_id_cols=X.var_id_cols)


		# Ys = Y[:,Js] # guaranteed to be equal to Y

		DXYs = compute(Diag(matrixproduct(Xs.matrix',Y.matrix)))

		DX2s = DX2[Is]

		dists[i:k:end] .= sqrt.(max.(0.0, DX2s .+ DY2 .- 2DXYs))
	end

	sparse(I,J,dists,N1,N2)
end
