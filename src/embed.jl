function embed_points(origData, origEmbedding, newData; k=10)
	N1 = size(origData,2)
	N2 = size(newData,2)
	@assert size(origEmbedding,2)==N1
	k = min(N1,k) # cannot have more neighbors than points!

    tree = BallTree(origData) # Choose KDTree if dimension is lower?
    indices,dists = knn(tree, newData, k)

    # N1xN2 sparse matrix where each column is the weights in the original data set for one point in the new dataset
    I = reduce(vcat,indices)
    J = repeat(1:N2; inner=k)

    adj = sparse(I,J,true,N1,N2)


    # standard weights
    w = (x->1.0./max.(1e-12,x.^2)).(dists) # weigth by 1/dist² but avoid div by zero

    # try to combat curse of dimensionality by giving closer points higher weights
    # w = (x->1.0./(x.-0.5*minimum(x)).^2).(dists)


    w ./= sum.(w) # normalize weights
    weights = reduce(vcat, w)

    # A = sparse(I,J,1/k,N1,N2) # equal weights
    A = sparse(I,J,weights,N1,N2)
    newEmbedding = origEmbedding*A

    adj, newEmbedding
end
