module UMAPExt

using ReproducibleJobs
using ReproducibleJobs: create_job, cached, TypeTag, Cache
using SingleCellProjections
using SingleCellProjections.Impl: DataMatrixFunction, Projectable, Action, Eval, Projection, Mat, Var, Obs, get_matrix_job, get_job, prefixed_ids_job

using DataFrames

using UMAP: UMAP, UMAPResult

using Random



# TODO: Can we do better than this? How do we handle breaking changes to the UMAPResult struct layout?
ReproducibleJobs.deduplicate_type(::Type{<:UMAPResult}) = true
ReproducibleJobs.deconstruct_type(::Type{<:UMAPResult}) = true
ReproducibleJobs.type_to_tag(::Type{<:UMAPResult}) = TypeTag(:UMAPResult)
ReproducibleJobs.tag_to_type(::Val{:UMAPResult}) = UMAPResult
ReproducibleJobs.deconstruct(r::UMAPResult) = (r.data, r.embedding, r.config, r.knns_dists, r.fs_sets, r.graph)
ReproducibleJobs.reconstruct(::Type{<:UMAPResult}, (data,embedding,config,knns_dists,fs_sets,graph)::Tuple) =
	UMAPResult(parent(data), parent(embedding), config, parent.(knns_dists), fs_sets, graph)


ReproducibleJobs.deduplicate_type(::Type{<:UMAP.UMAPConfig}) = false
ReproducibleJobs.deconstruct_weak_rec(x::UMAP.UMAPConfig) = x
ReproducibleJobs.reconstruct_weak_rec(x::UMAP.UMAPConfig) = x

function ReproducibleJobs.cache_save(::Cache, io, name, x::UMAP.UMAPConfig)
	io[name] = x # Rely on JLD2 standard handling for saving/loading
	nothing
end





function umap_model(matrix; ndim::Int, seed::Int, kwargs...)
	# Brittle workaround for improving UMAP reproducibility (see https://discourse.julialang.org/t/how-could-i-save-and-restore-the-status-of-random-seed/61941/4)
	rng_state = copy(Random.default_rng())
	Random.seed!(seed)
	res = UMAP.fit(matrix, ndim; kwargs...)
	copy!(Random.default_rng(), rng_state)
	res
end
umap_embedding(result::UMAPResult) = result.embedding

# umap_project(result::UMAPResult, matrix) = UMAP.transform(result, parent(matrix)).embedding
function umap_project(result::UMAPResult, matrix; seed::Int)
	# Brittle workaround for improving UMAP reproducibility (see https://discourse.julialang.org/t/how-could-i-save-and-restore-the-status-of-random-seed/61941/4)
	rng_state = copy(Random.default_rng())
	Random.seed!(seed)
	res = UMAP.transform(result, parent(matrix))
	copy!(Random.default_rng(), rng_state)
	res.embedding
end



function umap_impl(action::Action, matrix; ndim, seed, kwargs...)
	# First create UMAP model
	umap_model_job = cached(create_job(umap_model, matrix; ndim, seed, kwargs..., __version=v"0.3.0"))

	if action isa Eval
		return create_job(umap_embedding, umap_model_job; __version=v"0.3.0")
	else# if action isa Projection
		return cached(create_job(umap_project, umap_model_job, action(matrix); seed, __version=v"0.3.0"))
	end
end


function umap(::Mat, data; ndim, seed, kwargs...)
	matrix_job = get_matrix_job(data)
	create_job(Projectable(umap_impl), matrix_job; ndim, seed, kwargs...)
end
umap(::Obs, data; ndim, kwargs...) = get_job(Obs(), data)
umap(::Var, data; ndim, kwargs...) = prefixed_ids_job("id", "UMAP ", ndim)


function SingleCellProjections.umap(data; ndim, seed=1234, kwargs...)
	create_job(DataMatrixFunction(umap), data; ndim, seed, kwargs...)
end






end
