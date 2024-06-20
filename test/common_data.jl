# Paths
pbmc_path = joinpath(pkgdir(SingleCellProjections), "test/data/500_PBMC_3p_LT_Chromium_X_50genes")
h5_path = joinpath(pbmc_path, "filtered_feature_bc_matrix.h5")
mtx_path = joinpath(pbmc_path, "filtered_feature_bc_matrix/matrix.mtx.gz")

rna_adt_h5_path = joinpath(pkgdir(SingleCellProjections), "test/data/GSE164378_RNA_ADT_3P_P1_subsetted.h5")
rna_h5_path = joinpath(pkgdir(SingleCellProjections), "test/data/GSE164378_RNA_3P_P1_subsetted.h5")

# Ground truth
expected_mat = read_matrix(joinpath(pbmc_path,"expected_matrix.csv"))
expected_nnz = count(!iszero, expected_mat)
expected_feature_ids = vec(read_strings(joinpath(pbmc_path,"expected_feature_ids.csv")))
expected_barcodes = vec(read_strings(joinpath(pbmc_path,"expected_barcodes.csv")))

expected_feature_names = read_strings(joinpath(pbmc_path,"filtered_feature_bc_matrix/features.tsv.gz"),'\t')[:,2]
expected_feature_types = fill("Gene Expression", 50)
expected_feature_genome = fill("GRCh38", 50)

expected_sparse = sparse(expected_mat)
params = scparams(expected_sparse, DataFrame(id=expected_feature_ids, name=expected_feature_names, feature_type=expected_feature_types); use_cache=false)

# Data shared between tests
counts = load10x(h5_path)

counts.obs.group = rand(StableRNG(904), ("A","B","C"), size(counts,2))
counts.obs.value = 1 .+ randn(StableRNG(905), size(counts,2))

transformed = sctransform(counts; use_cache=false)
normalized = normalize_matrix(transformed, "group", "value")
reduced = svd(normalized; nsv=10, niter=4, rng=StableRNG(102))
