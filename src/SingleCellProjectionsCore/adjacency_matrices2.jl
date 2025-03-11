function knn_adjacency_matrix2(f, X, Y; k, tree_fun=KDTree)
    N1 = size(X,2)
    N2 = size(Y,2)
    @assert size(X,1) == size(Y,1)
    k = min(N1,k) # cannot have more neighbors than points!

    tree = tree_fun(X)
    indices,dists = knn(tree, Y, k)

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
