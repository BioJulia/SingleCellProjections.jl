# Data source
10x example file "500_PBMC_3p_LT_Chromium_X_filtered_feature_bc_matrix.h5" downloaded from here:
https://www.10xgenomics.com/resources/datasets/500-human-pbm-cs-3-lt-v-3-1-chromium-x-3-1-low-6-1-0

# Subsetting of genes
path/to/cellranger-6.1.2/bin/cellranger reanalyze --disable-ui --id 500_PBMC_3p_LT_Chromium_X_50genes --genes genes_50.csv --matrix 500_PBMC_3p_LT_Chromium_X_filtered_feature_bc_matrix.h5

The output files:
500_PBMC_3p_LT_Chromium_X_50genes/outs/filtered_feature_bc_matrix.h5
500_PBMC_3p_LT_Chromium_X_50genes/outs/filtered_feature_bc_matrix/matrix.mtx.gz
500_PBMC_3p_LT_Chromium_X_50genes/outs/filtered_feature_bc_matrix/barcodes.tsv.gz
500_PBMC_3p_LT_Chromium_X_50genes/outs/filtered_feature_bc_matrix/features.tsv.gz
where copied here.

# Create dense matrix for ground truth
path/to/cellranger-6.1.2/bin/cellranger mat2csv 500_PBMC_3p_LT_Chromium_X_50genes/outs/filtered_feature_bc_matrix.h5 dense.csv

dense.csv was then split manually and gzipped to produce
expected_barcodes.csv.gz
expected_feature_ids.csv.gz
expected_matrix.csv.gz



# Subsetting of barcodes
path/to/cellranger-9.0.1/bin/cellranger reanalyze --disable-ui --id 500_PBMC_3p_LT_Chromium_X_50genes_272barcodes --barcodes barcodes_272.csv --matrix filtered_feature_bc_matrix.h5

The output file 500_PBMC_3p_LT_Chromium_X_50genes_272barcodes/outs/filtered_feature_bc_matrix.h5
was renamed to
filtered_feature_bc_matrix_272barcodes.h5
and copied here.
