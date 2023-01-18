function knn_adjacency_matrix(X; k, make_symmetric=true)
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




_randinit(::Val{ndim}, rng, N::Int, scale) where ndim = scale.*randn(rng, SVector{ndim,Float64}, N)


# inefficient reference implementation O(N²)
function charge_forces_reference!(vel::AbstractVector, pos::AbstractVector; charge, charge_min_distance, alpha)
    N = length(pos)
    charge_min_distance2 = charge_min_distance^2
    # inefficient double loop (upper triangular part)
    for j=2:N
        for i=1:j-1
            u = pos[j]-pos[i] # vector between points
            denom = sum(abs2,u)
            denom<charge_min_distance2 && (denom = max(1e-9,charge_min_distance*sqrt(denom))) # limit force for points very close to each other
            Fc = alpha*charge/denom * u
            vel[j] += Fc
            vel[i] -= Fc
        end
    end
end



function charge_forces_rec!(vel::AbstractVector, pos::AbstractVector, pointInd, tree::BarnesHutTree{N1,N2}, depth, nodeInd, firstPointInd, nodeDiameter2, charge, charge_min_distance2, alpha, theta2) where {N1,N2}
    node = tree.nodes[nodeInd]
    # for each child
    childNodeInd = nodeInd+1
    for i=1:N2
        endPointInd = firstPointInd + node.childLengths[i]
        childRange = firstPointInd:endPointInd-1

        p = pos[pointInd]
        if depth+1>=tree.maxDepth || length(childRange)<=tree.leafSize
            # leaf, process all points
            for k in childRange
                pointInd2 = tree.pointIndices[k]
                pointInd==pointInd2 && continue # skip self
                u = p - pos[pointInd2]
                denom = sum(abs2,u)
                denom<charge_min_distance2 && (denom = max(1e-9,sqrt(charge_min_distance2*denom))) # limit force for points very close to each other
                vel[pointInd] += alpha*charge/denom * u
            end
        else
            # internal node
            childNode = tree.nodes[childNodeInd]
            u = p-childNode.centerOfGravity
            d2 = sum(abs2,u)

            # approximate?
            if nodeDiameter2 < theta2*d2
                denom = d2
                denom<charge_min_distance2 && (denom = max(1e-9,sqrt(charge_min_distance2*denom))) # limit force for points very close to each other
                vel[pointInd] += length(childRange)*alpha*charge/denom * u
            else
                # otherwise recurse
                charge_forces_rec!(vel, pos, pointInd, tree, depth+1, childNodeInd, firstPointInd, nodeDiameter2/4, charge, charge_min_distance2, alpha, theta2)
            end
            childNodeInd = childNode.skipPointer
        end


        firstPointInd = endPointInd
    end
end

# Barnes-Hut implementation
function charge_forces!(vel::AbstractVector, pos::AbstractVector, tree::BarnesHutTree; charge, charge_min_distance, alpha, theta)
    build!(tree, pos)

    charge_min_distance2 = charge_min_distance^2
    theta2 = theta^2

    # for i=1:length(pos) # each point against tree
    #     charge_forces_rec!(vel, pos, i, tree, 0, 1, 1, diameter2(tree), charge, charge_min_distance2, alpha, theta2)
    # end
    # each point against tree
    @sync for r in splitrange(1:length(pos), max(1,Threads.nthreads()-1))
        Threads.@spawn for i in r
            charge_forces_rec!(vel, pos, i, tree, 0, 1, 1, diameter2(tree), charge, charge_min_distance2, alpha, theta2)
        end
    end
end



function link_forces!(vel::AbstractVector, pos::AbstractVector, adj; link_distance, link_strength, alpha)
    N = length(pos)
    adjR = rowvals(adj)
    adjV = nonzeros(adj)
    for j=2:N
        for k in nzrange(adj,j)
            i = adjR[k]
            i>=j && break # only upper triangular part
            adjV[k]==false && continue # handle zeros that are not structural...
            u = (pos[j].+vel[j]) .- (pos[i].+vel[i])
            d = sqrt(sum(abs2,u))
            Fl = alpha*link_strength*(d-link_distance)/(2*d) * u
            vel[j] -= Fl
            vel[i] += Fl
        end
    end
end


# d3 inspired force layout
function force_layout(::Val{ndim}, adj::AbstractMatrix;
                      niter=100,
                      link_distance=4, link_strength=2,
                      charge=5, charge_min_distance=1, theta = 0.9,
                      center_strength=0.05,
                      velocity_decay=0.9,
                      initialAlpha = 1.0, finalAlpha = 1e-3,
                      initialScale = 10,
                      rng = Random.default_rng()) where ndim
    N = size(adj,1)
    @assert size(adj,2)==N
    @assert issymmetric(adj) # TODO: support upper triangular adj matrix too?

    @assert initialAlpha >= finalAlpha
    @assert finalAlpha > 0
    beta = -log(finalAlpha/initialAlpha)/niter

    pos = _randinit(Val(ndim), rng, N, initialScale)
    vel = zeros(SVector{ndim,Float64},N)

    tree = BarnesHutTree(ndim)

    for iter = 1:niter
        alpha = initialAlpha*exp(-beta*iter)

        charge != 0 && charge_forces!(vel, pos, tree; charge=charge, charge_min_distance=charge_min_distance, alpha=alpha, theta=theta)
        link_strength>0 && link_forces!(vel, pos, adj; link_distance=link_distance, link_strength=link_strength, alpha=alpha)

        # point updates
        center = @SVector zeros(ndim)
        for i=1:N
            # forces acting on single points
            vel[i] -= alpha*center_strength*pos[i] # link attaching each point to the center

            vel[i] *= velocity_decay
            pos[i] += vel[i]

            center += pos[i]
        end
        # center points
        center /= N
        for i=1:N
            pos[i] -= center
        end
    end

    reduce(hcat,pos)
end

force_layout(adj::AbstractMatrix; ndim::Int=2, kwargs...) =
    force_layout(Val(ndim), adj; kwargs...)
