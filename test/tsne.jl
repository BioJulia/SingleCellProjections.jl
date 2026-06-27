using Test
using SingleCellProjections
using ReproducibleJobs: fetch!, forward!
import TSne
using Random


tsne_reference_impl(matrix; ndim) =
	permutedims(TSne.tsne(matrix', ndim, 0; pca_init=true))

function run_tsne_tests()
	@testset "t-SNE" begin
		counts_job = SCP.load_counts(h5_path; sample_names="a")
		counts_sub_job = SCP.load_counts(h5_subset_path; sample_names="p")

		# TODO: test forwarding
		# TODO: test hash stability

		transformed_job = SCP.logtransform(counts_job)
		normalized_job = SCP.normalize_matrix(counts_job)
		pca_job = SCP.pca(normalized_job; nsv=10)

		pca_dm = fetch!(pca_job)

		tsne_job = SCP.tsne(pca_job; ndim=2)
		tsne_dm = fetch!(tsne_job)

		@test tsne_dm.obs === pca_dm.obs


		tsne_ans = tsne_reference_impl(pca_dm.matrix; ndim=2)


		# TODO: Update. What can we expect here???
		@test tsne_dm.matrix ≈ tsne_ans


		# TODO: Test projection
	end
end
