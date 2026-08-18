_randinit(::Val{ndim}, rng, N::Int, scale) where ndim = scale.*randn(rng, SVector{ndim,Float64}, N)


# inefficient reference implementation O(N²)
# This function is based on d3-force: https://github.com/d3/d3-force, also see LICENSE.md.
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


# This function is based on d3-force: https://github.com/d3/d3-force, also see LICENSE.md.
function charge_forces_rec!(vel::AbstractVector, pos::AbstractVector, point_ind, tree::BarnesHutTree{N1,N2}, depth, node_ind, first_point_ind, node_diameter2, charge, charge_min_distance2, alpha, theta2) where {N1,N2}
    node = tree.nodes[node_ind]
    # for each child
    child_node_ind = node_ind+1
    for i=1:N2
        end_point_ind = first_point_ind + node.child_lengths[i]
        child_range = first_point_ind:end_point_ind-1

        p = pos[point_ind]
        if depth+1>=tree.max_depth || length(child_range)<=tree.leaf_size
            # leaf, process all points
            for k in child_range
                point_ind2 = tree.point_indices[k]
                point_ind==point_ind2 && continue # skip self
                u = p - pos[point_ind2]
                denom = sum(abs2,u)
                denom<charge_min_distance2 && (denom = max(1e-9,sqrt(charge_min_distance2*denom))) # limit force for points very close to each other
                vel[point_ind] += alpha*charge/denom * u
            end
        else
            # internal node
            child_node = tree.nodes[child_node_ind]
            u = p-child_node.center_of_gravity
            d2 = sum(abs2,u)

            # approximate?
            if node_diameter2 < theta2*d2
                denom = d2
                denom<charge_min_distance2 && (denom = max(1e-9,sqrt(charge_min_distance2*denom))) # limit force for points very close to each other
                vel[point_ind] += length(child_range)*alpha*charge/denom * u
            else
                # otherwise recurse
                charge_forces_rec!(vel, pos, point_ind, tree, depth+1, child_node_ind, first_point_ind, node_diameter2/4, charge, charge_min_distance2, alpha, theta2)
            end
            child_node_ind = child_node.skip_pointer
        end


        first_point_ind = end_point_ind
    end
end

# Barnes-Hut implementation
# This function is based on d3-force: https://github.com/d3/d3-force, also see LICENSE.md.
function charge_forces!(vel::AbstractVector, pos::AbstractVector, tree::BarnesHutTree; charge, charge_min_distance, alpha, theta)
    build!(tree, pos)

    charge_min_distance2 = charge_min_distance^2
    theta2 = theta^2

    # for i=1:length(pos) # each point against tree
    #     charge_forces_rec!(vel, pos, i, tree, 0, 1, 1, diameter2(tree), charge, charge_min_distance2, alpha, theta2)
    # end
    # each point against tree
    # @sync for r in splitrange(1:length(pos), max(1,Threads.nthreads()-1))
    #     Threads.@spawn for i in r
    #         charge_forces_rec!(vel, pos, i, tree, 0, 1, 1, diameter2(tree), charge, charge_min_distance2, alpha, theta2)
    #     end
    # end
    # tforeach(1:length(pos)) do i
    tforeach(1:length(pos); scheduler=:greedy, chunking=true, minchunksize=128) do i # TODO: Revisit parameters
        charge_forces_rec!(vel, pos, i, tree, 0, 1, 1, diameter2(tree), charge, charge_min_distance2, alpha, theta2)
    end
end


# This function is based on d3-force: https://github.com/d3/d3-force, also see LICENSE.md.
function link_forces!(vel::AbstractVector, pos::AbstractVector, adj; link_distance, link_strength, alpha)
    N = length(pos)
    adj_r = rowvals(adj)
    adj_v = nonzeros(adj)
    for j=2:N
        for k in nzrange(adj,j)
            i = adj_r[k]
            i>=j && break # only upper triangular part
            adj_v[k]==false && continue # handle zeros that are not structural?
            u = (pos[j].+vel[j]) .- (pos[i].+vel[i])
            d = sqrt(sum(abs2,u))
            Fl = alpha*link_strength*(d-link_distance)/(2*d) * u
            vel[j] -= Fl
            vel[i] += Fl
        end
    end
end


# This function is based on d3-force: https://github.com/d3/d3-force, also see LICENSE.md.
# NB: If we change default values of kwargs here, documentation should be updated in reduce.jl as well.
function force_layout(::Val{ndim}, adj::AbstractMatrix;
                      niter=100,
                      link_distance=4, link_strength=2,
                      charge=5, charge_min_distance=1, theta = 0.9,
                      center_strength=0.05,
                      velocity_decay=0.9,
                      initial_alpha = 1.0, final_alpha = 1e-3,
                      initial_scale = 10,
                      seed = nothing,
                      rng = seed !== nothing ? seed2rng(seed) : Random.default_rng(),
                      progress = nothing,
                      tick = nothing) where ndim
    N = size(adj,1)
    @assert size(adj,2)==N
    @assert issymmetric(adj) # TODO: support upper triangular adj matrix too?

    @assert initial_alpha >= final_alpha
    @assert final_alpha > 0
    beta = -log(final_alpha/initial_alpha)/niter

    pos = _randinit(Val(ndim), rng, N, initial_scale)
    vel = zeros(SVector{ndim,Float64},N)

    tree = BarnesHutTree(ndim)

    isnothing(tick) || tick()
    isnothing(progress) || progress(niter) # initialize

    for iter = 1:niter
        alpha = initial_alpha*exp(-beta*iter)

        charge != 0 && charge_forces!(vel, pos, tree; charge=charge, charge_min_distance=charge_min_distance, alpha=alpha, theta=theta)
        isnothing(tick) || tick()
        link_strength>0 && link_forces!(vel, pos, adj; link_distance=link_distance, link_strength=link_strength, alpha=alpha)
        isnothing(tick) || tick()

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

        isnothing(tick) || tick()
        isnothing(progress) || progress() # step
    end

    reduce(hcat,pos)
end

force_layout(adj::AbstractMatrix; ndim::Int=3, kwargs...) =
    force_layout(Val(ndim), adj; kwargs...)
