isgz(fn) = lowercase(splitext(fn)[2])==".gz"
_open(f, fn) = open(fn) do io
    f(isgz(fn) ? GzipDecompressorStream(io) : io)
end

read_matrix(fn,delim=',') = _open(io->readdlm(io,delim,Int), fn)
read_strings(fn,delim=',') = _open(io->readdlm(io,delim,String), fn)


@testset "Basic Workflow" begin
	pbmc_path = joinpath(pkgdir(SingleCellProjections), "test/data/500_PBMC_3p_LT_Chromium_X_50genes")
	h5_path = joinpath(pbmc_path, "filtered_feature_bc_matrix.h5")
	mtx_path = joinpath(pbmc_path, "filtered_feature_bc_matrix/matrix.mtx.gz")

    expected_mat = read_matrix(joinpath(pbmc_path,"expected_matrix.csv"))
    expected_nnz = count(!iszero, expected_mat)
    expected_feature_ids = vec(read_strings(joinpath(pbmc_path,"expected_feature_ids.csv")))
    expected_barcodes = vec(read_strings(joinpath(pbmc_path,"expected_barcodes.csv")))

    expected_feature_names = read_strings(joinpath(pbmc_path,"filtered_feature_bc_matrix/features.tsv.gz"),'\t')[:,2]
    expected_feature_types = fill("Gene Expression", 50)
    expected_feature_genome = fill("GRCh38", 50)


	@testset "load10x $(split(basename(p),'.';limit=2)[2]) lazy=$lazy" for p in (h5_path,mtx_path), lazy in (false, true)
		counts = load10x(p; lazy)
		@test size(counts)==(50,587)
		@test nnz(counts.matrix) == expected_nnz

		@test Set(names(counts.obs)) == Set(("id", "barcode"))
		@test counts.obs.id == expected_barcodes
		@test counts.obs.barcode == expected_barcodes

		if p==h5_path
			@test Set(names(counts.var)) == Set(("id", "name", "feature_type", "genome"))
			@test counts.var.genome == expected_feature_genome
		else
			@test Set(names(counts.var)) == Set(("id", "name", "feature_type"))
		end
		@test counts.var.id == expected_feature_ids
		@test counts.var.name == expected_feature_names
		@test counts.var.feature_type == expected_feature_types

		@test counts.obs_id_cols == ["id"]
		@test counts.var_id_cols == ["id", "feature_type"]

		if lazy
			@test counts.matrix.filename == p
			counts = load_counts(counts)
		end

		@test counts.matrix == expected_mat
		@test counts.matrix isa SparseMatrixCSC{Int64,Int32}
	end


	@testset "load_counts $(split(basename(p),'.';limit=2)[2]) lazy=$lazy lazy_merge=$lazy_merge" for p in (h5_path,mtx_path), lazy in (false, true), lazy_merge in (false, true)
		counts = load_counts([p,p]; sample_names=["a","b"], lazy, lazy_merge)

		@test size(counts)==(50,587*2)
		@test nnz(counts.matrix) == expected_nnz*2

		@test Set(names(counts.obs)) == Set(("id", "sampleName", "barcode"))
		@test counts.obs.id == [string.("a_",expected_barcodes); string.("b_",expected_barcodes)]
		@test counts.obs.sampleName == [fill("a",587); fill("b",587)]
		@test counts.obs.barcode == [expected_barcodes; expected_barcodes]

		if p==h5_path
			@test Set(names(counts.var)) == Set(("id", "name", "feature_type", "genome"))
			@test counts.var.genome == expected_feature_genome
		else
			@test Set(names(counts.var)) == Set(("id", "name", "feature_type"))
		end
		@test counts.var.id == expected_feature_ids
		@test counts.var.name == expected_feature_names
		@test counts.var.feature_type == expected_feature_types

		@test counts.obs_id_cols == ["id"]
		@test counts.var_id_cols == ["id", "feature_type"]

		if lazy_merge
			counts = load_counts(counts)
		end

		@test counts.matrix == [expected_mat expected_mat]
		@test counts.matrix isa SparseMatrixCSC{Int64,Int32}
	end

	# TODO: load_counts with user-provided load function

	counts = load10x(h5_path)

	# filter
	# annotation stuff

	# sctransform
	# logtransform
	# tf_idf_transform
	# normalize (incl. center, scale, categorical regression, linear regression)
	# svd
	# force-layout



end
