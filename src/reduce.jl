"""
    SCP.svd(data; nsv, seed=1234, kwargs...) -> Job

Compute a truncated SVD of `data`, keeping `nsv` singular values. Returns a `DataMatrix`
containing the SVD result. Uses a randomized algorithm based on
Halko, Martinsson, and Tropp (2011).

Keyword arguments controlling the iterative procedure:
- `seed` — random seed for reproducibility.
- `subspacedims` — dimension of the random subspace (default `4nsv`).
- `niter` — number of power iterations (default `3`).

See also [`pca`](@ref), [`loadings`](@ref).
"""
function svd(matrix; nsv, seed=1234, kwargs...)
	create_job(DataMatrixFunction(Impl.svd), matrix; nsv, seed, kwargs...)
end


"""
    SCP.pca(data; nsv, seed=1234, kwargs...) -> Job

Compute PCA of `data`, keeping `nsv` principal components. Returns a `DataMatrix` where
the variables are the principal components and the observations are unchanged. Uses a
randomized SVD algorithm based on Halko, Martinsson, and Tropp (2011).

The returned principal components are scaled by the singular values, to make this an accurate
`nsv`-dimensional approximation of the original data.

Keyword arguments controlling the iterative procedure:
- `seed` — random seed for reproducibility.
- `subspacedims` — dimension of the random subspace (default `4nsv`).
- `niter` — number of power iterations (default `3`).

# Examples

Compute a 100-dimensional PCA of `normalized`.
```julia
julia> SCP.pca(normalized; nsv=100)
```

See also [`svd`](@ref), [`loadings`](@ref), [`normalize_matrix`](@ref).
"""
function pca(data; nsv, seed=1234, kwargs...)
	create_job(DataMatrixFunction(Impl.pca), data; nsv, seed, kwargs...)
end


"""
    SCP.loadings(data; nsv, seed=1234, kwargs...) -> Job

Extract PCA loadings from `data`. Returns a `DataMatrix` where each column is a loading
vector. The loadings are not affected by projection. Uses the same randomized SVD algorithm
as `SCP.pca` and accepts the same keyword arguments (`nsv`, `seed`, `subspacedims`, `niter`).

# Examples

Compute the loadings of `normalized` for the 100 first principal components.
Useful in combination with a call to `SCP.pca` (with the same parameters).
```julia
julia> SCP.loadings(normalized; nsv=100)
```

See also [`pca`](@ref), [`svd`](@ref).
"""
function loadings(args...; nsv, seed=1234, kwargs...)
	create_job(DataMatrixFunction(Impl.loadings), args...; nsv, seed, kwargs...)
end


"""
    SCP.force_layout(data; ndim=3, kwargs...) -> Job

Compute a force-directed layout embedding of `data`. Returns a `DataMatrix` with `ndim`
layout dimensions as variables.

Keyword arguments:
- `k` — number of nearest neighbors for the graph.
- `k_fraction` — alternative to `k`, specify neighbors as a fraction of observations.
- `niter` — number of force simulation iterations (default `100`).
- `link_distance`, `link_strength` — link force parameters (defaults `40`, `0.05`).
- `charge`, `charge_min_distance`, `theta` — repulsion parameters (defaults `40`, `1`, `0.9`).
- `center_strength` — centering force (default `0.05`).
- `velocity_decay` — velocity damping (default `0.9`).
- `initialAlpha`, `finalAlpha` — simulation temperature schedule (defaults `1.0`, `1e-3`).
- `initialScale` — initial coordinate scale (default `10`).
- `seed` — random seed (default `1234`).
- `k_projection` — neighbors used when projecting onto this layout (default `10`).

# Examples

```julia
julia> SCP.force_layout(reduced; ndim=3, seed=4567, k=100, k_projection=25)
```

See also [`transform_coords`](@ref), [`find_optimal_coord_transform`](@ref), [`umap`](@ref), [`tsne`](@ref).
"""
function force_layout(args...; ndim=3, kwargs...)
	create_job(DataMatrixFunction(Impl.force_layout), args...; ndim, kwargs...)
end
