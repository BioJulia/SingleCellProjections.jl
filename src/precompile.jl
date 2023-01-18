using SnoopPrecompile

@precompile_setup begin
	base_path = joinpath(pkgdir(SingleCell10x)) # TODO: copy file to this repo?
	h5_path = joinpath(base_path, "test/data/500_PBMC_3p_LT_Chromium_X_50genes/filtered_feature_bc_matrix.h5")
	@precompile_all_calls begin
		counts = load10x(h5_path)
		counts2 = load_counts([h5_path,h5_path]; sample_names=["a","b"])
		transformed = sctransform(counts; use_cache=false, verbose=false)
		centered = normalize_matrix(transformed)
		reduced = svd(centered; nsv=10)
		fl = force_layout(reduced; ndim=3, k=100)
	end
end