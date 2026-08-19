using Test
using SingleCellProjections
using ReproducibleJobs: fetch!, forward!
import TSne
using Random


tsne_reference_impl(matrix; ndim, max_iter) =
	permutedims(TSne.tsne(matrix', ndim, 0, max_iter; pca_init=true, progress=false))

function run_tsne_tests()
	@testset "t-SNE" begin
		counts_job = SCP.load_counts(h5_path; sample_names="a")

		# t-SNE scales as O(ncells^2) so subset to speed up tests.
		counts_job = SCP.filter_obs(1:100, counts_job)

		# TODO: test forwarding
		# TODO: test hash stability

		transformed_job = SCP.logtransform(counts_job)
		normalized_job = SCP.normalize_matrix(counts_job)
		pca_job = SCP.pca(normalized_job; nsv=10)

		pca_dm = fetch!(pca_job)

		tsne_job = SCP.tsne(pca_job; ndim=2, max_iter=250)
		tsne_dm = fetch!(tsne_job)

		@test tsne_dm.obs === pca_dm.obs


		tsne_ans = tsne_reference_impl(pca_dm.matrix; ndim=2, max_iter=250)


		# TODO: Update. What can we expect here???
		@test tsne_dm.matrix ≈ tsne_ans


		@testset "Unsupported kwargs" begin
			@test_throws "pca_init kwarg is not supported" SCP.tsne(pca_job; ndim=2, pca_init=false)
			@test_throws "distance kwarg is not supported" SCP.tsne(pca_job; ndim=2, distance=true)
			@test_throws "extended_output kwarg is not supported" SCP.tsne(pca_job; ndim=2, extended_output=true)
		end

		# TODO: Test projection
	end
end
