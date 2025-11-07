@testset "load_counts" begin
	P,N = (50,587)

	# TODO: Test .mtx file (implement with specs first!)
	counts_job = Jobs.load_counts(h5_path; sample_names="a")
	
	# Test result
	let counts = fetch!(counts_job)
		@test size(counts)==(P,N)
		@test nnz(counts.matrix) == expected_nnz

		@test names(counts.obs) == ["cell_id", "sample_name", "barcode"]
		@test counts.obs.cell_id == string.("a_",expected_barcodes)
		@test counts.obs.sample_name == fill("a",N)
		@test counts.obs.barcode == expected_barcodes

		@test names(counts.var) == ["id", "name", "feature_type", "genome"]
		@test counts.var.id == expected_feature_ids
		@test counts.var.name == expected_feature_names
		@test counts.var.feature_type == expected_feature_types
		@test counts.var.genome == expected_feature_genome

		@test counts.matrix == expected_mat
		@test counts.matrix isa SparseMatrixCSC{Int64,Int32}
	end

end
