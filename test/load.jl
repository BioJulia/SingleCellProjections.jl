@testset "Load" begin
	P,N = (50,587)

	@testset "load10x $(split(basename(p),'.';limit=2)[2]) lazy=$lazy" for p in (h5_path,mtx_path), lazy in (false, true)
		counts = load10x(p; lazy)

		@test size(counts)==(P,N)
		@test nnz(counts.matrix) == expected_nnz

		@test Set(names(counts.obs)) == Set(("barcode",))
		@test counts.obs.barcode == expected_barcodes

		matrix_name = lazy ? "Lazy10xMatrix" : "SparseMatrixCSC"
		if p==h5_path
			@test Set(names(counts.var)) == Set(("id", "name", "feature_type", "genome"))
			@test counts.var.genome == expected_feature_genome
			test_show(counts; matrix=matrix_name, var=["id", "feature_type", "name", "genome"], obs=["barcode"], models="")
		else
			@test Set(names(counts.var)) == Set(("id", "name", "feature_type"))
			test_show(counts; matrix=matrix_name, var=["id", "feature_type", "name"], obs=["barcode"], models="")
		end
		@test counts.var.id == expected_feature_ids
		@test counts.var.name == expected_feature_names
		@test counts.var.feature_type == expected_feature_types

		@test names(counts.obs,1) == ["barcode"]
		@test names(counts.var,1) == ["id"]

		if lazy
			@test counts.matrix.filename == p
			counts = load_counts(counts)
		end

		@test counts.matrix == expected_mat
		@test counts.matrix isa SparseMatrixCSC{Int64,Int32}
	end

	@testset "load_counts $(split(basename(p),'.';limit=2)[2]) lazy=$lazy lazy_merge=$lazy_merge" for p in (h5_path,mtx_path), lazy in (false, true), lazy_merge in (false, true)
		@testset "2 samples" begin
			counts = load_counts([p,p]; sample_names=["a","b"], lazy, lazy_merge)

			@test size(counts)==(P,N*2)
			@test nnz(counts.matrix) == expected_nnz*2

			@test Set(names(counts.obs)) == Set(("cell_id", "sampleName", "barcode"))
			@test counts.obs.cell_id == [string.("a_",expected_barcodes); string.("b_",expected_barcodes)]
			@test counts.obs.sampleName == [fill("a",N); fill("b",N)]
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

			@test names(counts.obs,1) == ["cell_id"]
			@test names(counts.var,1) == ["id"]

			if lazy_merge
				counts = load_counts(counts)
			end

			@test counts.matrix == [expected_mat expected_mat]
			@test counts.matrix isa SparseMatrixCSC{Int64,Int32}
		end

		@testset "1 sample" begin
			counts = load_counts(p; sample_names="a", lazy, lazy_merge)
			counts2 = load_counts([p]; sample_names=["a"], lazy, lazy_merge)
			counts3 = load_counts(p; lazy, lazy_merge)
			counts4 = load_counts([p]; lazy, lazy_merge)

			@test counts == counts2
			@test counts3 == counts4

			@test counts.matrix == counts3.matrix
			@test counts.var == counts3.var


			@test size(counts)==(P,N)
			@test nnz(counts.matrix) == expected_nnz

			@test Set(names(counts.obs)) == Set(("cell_id", "sampleName", "barcode"))
			@test counts.obs.cell_id == string.("a_",expected_barcodes)
			@test counts.obs.sampleName == fill("a",N)
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

			@test names(counts.obs,1) == ["cell_id"]
			@test names(counts.var,1) == ["id"]

			if lazy_merge
				counts = load_counts(counts)
			end

			@test counts.matrix == expected_mat
			@test counts.matrix isa SparseMatrixCSC{Int64,Int32}
		end
	end

	# TODO: load_counts with user-provided load function
end
