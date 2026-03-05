function knn_adjacency_matrix2(f, X, Y; k, tree_fun=KDTree)
	N1 = size(X,2)
	N2 = size(Y,2)
	@assert size(X,1) == size(Y,1)
	k = min(N1,k) # cannot have more neighbors than points!

	tree = tree_fun(X)
	indices,dists = knn(tree, Y, k) # TODO: Threading?

	# N1xN2 sparse matrix where each column is the weights in the original data set for one point in the new dataset
	I = reduce(vcat,indices)
	J = repeat(1:N2; inner=k)

	if f === nothing
		weights = true
	else
		# standard weights
		# w = (x->1.0./max.(1e-12,x.^2)).(dists) # weigth by 1/dist² but avoid div by zero
		# w = f.(dists)
		# w = (x->f.(x)).(dists)
		# w ./= sum.(w) # normalize weights

		weights = map(dists) do d
			w = f.(d)
			w ./= sum(w) # normalize weights
		end


		weights = reduce(vcat, weights)
	end

	A = sparse(I,J,weights,N1,N2)
end

knn_adjacency_matrix2(X, Y; kwargs...) = knn_adjacency_matrix2(nothing, X, Y; kwargs...)




"""
	find_nearest_neighbors(X; k, tree_fun=KDTree)

Find the `k` nearest neighbors of each point in `X`. `tree_fun` can be used to choose which spatial data structure to use.

Returns a pair of `k×N` matrices (where `N` is the number of points (columns) of `X`):
* `indices`
* `dists`
"""
function find_nearest_neighbors(X; k, tree_fun=KDTree)
	N = size(X,2)
	k = min(N-1, k) # Cannot have more neighbors than points! (Excluding the point itself.)

	tree = tree_fun(X) # This is parallelized by NearestNeighbors.jl by default

	# Single-threaded version
	indices,dists = allknn(tree, k) # NB: This exludes the points themselves

	# TODO: Threaded version

	# From vectors of vectors to matrices
	indices = reduce(hcat, indices)
	dists = reduce(hcat, dists)
	indices, dists # k × size(X,2) matrices
end


"""
	find_nearest_neighbors(X, Y; k, tree_fun=KDTree)

Find the `k` nearest neighbors in `X` of each point in `Y`. `tree_fun` can be used to choose which spatial data structure to use.

Returns a pair of `k×N` matrices (where `N` is the number of points (columns) of `Y`):
* `indices`
* `dists`
"""
function find_nearest_neighbors(X, Y; k, tree_fun=KDTree)
	N1 = size(X,2)
	N2 = size(Y,2)
	@assert size(X,1) == size(Y,1) # Must have the same (number of) variables
	k = min(N1,k) # Cannot have more neighbors than points!

	tree = tree_fun(X) # This is parallelized by NearestNeighbors.jl by default

	# # Single-threaded version
	# indices,dists = knn(tree, Y, k)

	# Threaded version
	nt = max(1, Threads.nthreads()-1) # TODO: What's a good choice?
	c = chunks(1:N2; n=nt)
	results = tmap(c) do chunk
		knn(tree, @view(Y[:,chunk]), k)
	end
	# Combine results from each chunk
	indices = reduce(vcat, first.(results))
	dists = reduce(vcat, last.(results))

	# From vectors of vectors to matrices
	indices = reduce(hcat, indices)
	dists = reduce(hcat, dists)
	indices, dists # k × size(Y,2) matrices
end


function adjacency_matrix(indices; make_symmetric) # TODO: support normalize_weights::Bool kwarg (which is not allowed together with make_symmetric)?
	N = size(indices,2)
	k = size(indices,1)

	I = vec(indices)
	J = repeat(1:N; inner=k)

	adj = sparse(I,J,true,N,N)
	if make_symmetric
		adj = adj.|adj'
	end
	adj
end



"""
	weighted_adjacency_matrix(f, indices, dists; NX=size(indices,2), normalize_weights=true)

Construct a weighted adjacency matrix.
* `f(x)` - convert a distance `x` to a weight.
* `indices` - A `k×NY` matrix where each column contains the indices of the `k` nearest neighbors for that point.
* `dists` - A `k×NY` matrix where each entry contains the distances to the corresponding point in `indices`.

Optional kwargs:
* `NX` - If indices corresponds to a set of indices in *another* data set, `NX` must be used to specify the number of points in that data set.
* `normalize_weights` - If true (which is the default), the weights will be normalized to sum to one in each column.

Returns a `NX×NY` sparse adjacency matrix with non-zeroes set to the given weights.
"""
function weighted_adjacency_matrix(f, indices, dists; NX=size(indices,2), normalize_weights=true)
	@assert size(indices) == size(dists)

	NY = size(indices,2)
	k = size(indices,1)

	I = reduce(vcat, indices)
	J = repeat(1:NY; inner=k)

	if normalize_weights
		weights = float.(f.(dists)) # floats are required to do the division in-place
		weights ./= sum(weights; dims=1)
	else
		weights = f.(dists)
	end
	weights = reduce(vcat, weights)

	A = sparse(I, J, weights, NX, NY)
end
