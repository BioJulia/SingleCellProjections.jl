"""
	SVDModel <: ProjectionModel

A model used for projecting onto an `SVD` object.
Normally created using `svd(::DataMatrix)`.

See also: [`svd`](@ref)
"""
struct SVDModel <: ProjectionModel
	U::Matrix{Float64}
	S::Vector{Float64}
	var_match::DataFrame
	var::Symbol
	obs::Symbol
end

projection_isequal(m1::SVDModel, m2::SVDModel) = m1.U == m2.U && m1.S == m2.S && m1.var_match == m2.var_match


update_model(m::SVDModel; var=m.var, obs=m.obs, kwargs...) = (SVDModel(m.U, m.S, m.var_match, var, obs), kwargs)


"""
	svd(data::DataMatrix; nsv=3, var=:copy, obs=:copy, kwargs...)

Compute the Singular Value Decomposition (SVD) of `data` using the Random Subspace SVD algorithm from [Halko et al. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions"].
SVD is often used to perform Principal Component Analysis (PCA), which assumes that the data is centered.

* `nsv` - The number of singular values.
* `var` - Can be `:copy` (make a copy of source `var`) or `:keep` (share the source `var` object).
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

Additional kwargs related to numerical precision are passed to `SingleCellProjections.implicitsvd`.

See also: [`SingleCellProjections.implicitsvd`](@ref)
"""
function LinearAlgebra.svd(data::DataMatrix; nsv=3, var=:copy, obs=:copy, kwargs...)
	F = implicitsvd(data.matrix; nsv=nsv, kwargs...)
	model = SVDModel(F.U, F.S, select(data.var,1), var, obs)
	update_matrix(data, F, model; model.var, model.obs)
end


function project_impl(data::DataMatrix, model::SVDModel; verbose=true, kwargs...)
	@assert table_cols_equal(data.var, model.var_match) "SVD projection expects model and data variables to be identical."

	U = model.U
	S = model.S
	X = data.matrix

	V = X'U # TODO: compute F.U'X instead to get Vt directly
	V ./= max.(S,1e-100)' # To avoid NaNs if any singular value is zero
	matrix = SVD(U,S,Matrix(V'))
	update_matrix(data, matrix, model; model.obs, model.var)
end


struct NearestNeighborModel <: ProjectionModel
	name::String
	pre::Matrix{Float64}
	post::Matrix{Float64}
	var_match::DataFrame
	k::Int
	var::String
	obs::Symbol
end
NearestNeighborModel(name, pre::DataMatrix, post; k, var, obs) =
	NearestNeighborModel(name, obs_coordinates(pre), post, select(pre.var,1), k, var, obs)
NearestNeighborModel(name, pre::DataMatrix, post::DataMatrix; kwargs...) =
	NearestNeighborModel(name, pre, obs_coordinates(post); kwargs...)

function projection_isequal(m1::NearestNeighborModel, m2::NearestNeighborModel)
	m1.name == m2.name && m1.pre == m2.pre && m1.post == m2.post &&
	m1.var_match == m2.var_match && m1.k == m2.k
end

update_model(m::NearestNeighborModel; k=m.k, var=m.var, obs=m.obs, kwargs...) =
	(NearestNeighborModel(m.name, m.pre, m.post, m.var_match, k, var, obs), kwargs)


function project_impl(data::DataMatrix, model::NearestNeighborModel; adj_out=nothing, verbose=true, kwargs...)
	@assert table_cols_equal(data.var, model.var_match) "Nearest Neighbor projection expects model and data variables to be identical."

	adj, matrix = embed_points(model.pre, model.post, obs_coordinates(data); model.k)
	adj_out !== nothing && (adj_out[] = adj)

	update_matrix(data, matrix, model; model.var, model.obs)
end


"""
    force_layout(data::DataMatrix;
                 ndim=3,
                 k,
                 adj,
                 kprojection=10,
                 obs=:copy,
                 adj_out,
                 niter = 100,
                 link_distance = 4,
                 link_strength = 2,
                 charge = 5,
                 charge_min_distance = 1,
                 theta = 0.9,
                 center_strength = 0.05,
                 velocity_decay = 0.9,
                 initialAlpha = 1.0,
                 finalAlpha = 1e-3,
                 initialScale = 10,
                 seed,
                 rng)

Compute the Force Layout (also known as a force directed knn-graph or SPRING plots) for `data`.
Usually, `data` is a DataMatrix after reduction to `10-100` dimensions by `svd`.

A Force Layout is computed by running a physics simulation were the observations are connected by springs (such that connected observations are attracted), a general "charge" force repelling all observations from each other and a centering force that keep the observations around the origin.
The implementation is based on d3-force: https://github.com/d3/d3-force, also see LICENSE.md.

Exactly one of the kwargs `k` and `adj` must be provided. See details below.

General parameters:
* `k` - Number of nearest neighbors to connect each observation to (computes `adj` below).
* `adj` - An sparse, symmetric, adjacency matrix with booleans. `true` if two observations are connected by a spring and `false` otherwise.
* `kprojection` - The number of nearest neighbors used when projecting onto the resulting force layout. (Not used in the computation of the layout, only during projection.)
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).
* `adj_out` - Optional `Ref`. If specified, the (computed) `adj` matrix will be assigned to `adj_out`.

Paramters controlling the physics simulation:
* `niter` - Number of iterations to run the simulation.
* `link_distance` - The length of each spring.
* `link_strength` - The strength of the spring force.
* `charge` - The strength of the charge force.
* `charge_min_distance` - Used to avoid numerical instabilities by limiting the charge force for observations that are very close.
* `theta` - Parameter controlling accuracy in the Barnes-Hut approximation for charge forces.
* `center_strength` - Strength of the centering force.
* `velocity_decay` - At each iteration, the current velocity for an observations is multiplied by `velocity_decay`.
* `initialAlpha` - The alpha value decreases over time and allows larger changes to happen early, while being more stable towards the end.
* `finalAlpha` - See `initialAlpha`
* `initialScale` - The simulation is initialized by randomly drawing each observation from a multivariate Gaussian, and is scaled by `initialScale`.
* `seed` - Optional random seed used to init `rng`. NB: This requires the package `StableRNGs` to be loaded.
* `rng` - Optional RNG object. Useful for reproducibility.

# Examples

```julia
julia> force_layout(data; ndim=3, k=100)
```
"""
function force_layout(data::DataMatrix; ndim=3, k=nothing, adj=nothing, kprojection=10, obs=:copy, adj_out=nothing, kwargs...)
	@assert isnothing(k) != isnothing(adj) "Must specify exactly one of k and adj."
	if k !== nothing
		adj = knn_adjacency_matrix(obs_coordinates(data); k, make_symmetric=true)
	end
	adj_out !== nothing && (adj_out[] = adj)

	fl = force_layout(adj; ndim, kwargs...)
	model = NearestNeighborModel("force_layout", data, fl; k=kprojection, var="Force Layout Dim ", obs)
	update_matrix(data, fl, model; model.var, model.obs)
end


# - show -
function Base.show(io::IO, ::MIME"text/plain", model::SVDModel)
	print(io, "SVDModel(nsv=", length(model.S), ')')
end
function Base.show(io::IO, ::MIME"text/plain", model::NearestNeighborModel)
	print(io, "NearestNeighborModel(base=\"", model.name, "\", k=", model.k, ')')
end
