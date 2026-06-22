using Test
using SingleCellProjections
using ReproducibleJobs: fetch!, forward!
using UMAP
using Random


function umap_reference_impl(matrix; ndim, seed)
	rng_state = copy(Random.default_rng())
	Random.seed!(seed)
	res = UMAP.fit(matrix, ndim)
	copy!(Random.default_rng(), rng_state)
	res
end

function umap_transform_reference_impl(res, matrix; seed)
	rng_state = copy(Random.default_rng())
	Random.seed!(seed)
	res = UMAP.transform(res, matrix)
	copy!(Random.default_rng(), rng_state)
	res.embedding
end


function run_umap_tests()
	@testset "UMAP" begin
		counts_job = Jobs.load_counts(h5_path; sample_names="a")
		counts_sub_job = Jobs.load_counts(h5_subset_path; sample_names="p")

		# TODO: test forwarding
		# TODO: test hash stability

		transformed_job = Jobs.logtransform(counts_job)
		normalized_job = Jobs.normalize_matrix(counts_job)
		pca_job = Jobs.pca(normalized_job; nsv=10)

		pca_dm = fetch!(pca_job)

		umap_job = Jobs.umap(pca_job; ndim=2, seed=9876)
		umap_dm = fetch!(umap_job)

		@test umap_dm.obs === pca_dm.obs


		umap_result = umap_reference_impl(pca_dm.matrix; ndim=2, seed=9876)


		# TODO: Update. What can we expect here???
		@test umap_dm.matrix ≈ umap_result.embedding # If this fails, it's probably due to numerical differences due to threading in NearestNeighborDescent.jl. So we might need to relax this. Or, better, fix the instability somehow.
		# @test size(umap_dm.matrix) == size(umap_result.embedding)
		# @test typeof(parent(umap_dm.matrix)) == typeof(umap_result.embedding)


		umap_proj_job = Jobs.project(umap_job, counts_job=>counts_sub_job)
		umap_proj_dm = fetch!(umap_proj_job)

		Y = fetch!(Jobs.project(Jobs.get_matrix(pca_job), counts_job=>counts_sub_job))
		umap_proj_ans = umap_transform_reference_impl(umap_result, Y; seed=9876)
		@test umap_proj_dm.matrix ≈ umap_proj_ans

	end
end
