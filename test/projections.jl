@testset "Projections" begin
	proj_obs_ind = 1:12:size(counts,2)
	counts_proj = counts[:,proj_obs_ind]

	fl = force_layout(reduced; ndim=3, k=10, rng=StableRNG(408))

	@testset "from" begin
		@test_throws ArgumentError project(counts_proj,transformed)
		t2 = project(counts_proj, transformed; from=counts)
		@test materialize(t2) ≈ materialize(transformed)[:,proj_obs_ind] rtol=1e-3

		l2 = logtransform(counts_proj)
		@test_throws ArgumentError project(l2,normalized)
		n2 = project(l2,normalized; from=transformed)
		@test size(n2) == (size(normalized,1),size(l2,2)) # TODO: test the result more properly?
	end

	@testset "models" begin
		fl_proj = project(counts_proj, fl.models)
		@test materialize(fl_proj)≈materialize(fl)[:,proj_obs_ind] rtol=1e-5
	end

	@testset "Gene sets subset=$subset_genes rename=$rename_genes shuffle=$shuffle_genes" for subset_genes in (false,true), rename_genes in (false,true), shuffle_genes in (false,true)
		if subset_genes
			gene_ind = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49]
		else
			gene_ind = 1:size(expected_sparse,1)
		end
		new_gene_ids = copy(expected_feature_ids)
		rename_genes && (new_gene_ids[1:4:end] .= string.("NEW_", new_gene_ids[1:4:end]))
		ref_var = DataFrame(id=new_gene_ids, feature_type=expected_feature_types)[gene_ind,:]
		ref_mat = expected_sparse[gene_ind,:]

		counts_proj2 = counts[gene_ind,:]
		counts_proj2.var.id .= ref_var.id
		if shuffle_genes
			counts_proj2 = counts_proj2[randperm(StableRNG(498), length(gene_ind)), :]
		end
		empty!(counts_proj2.models)


		ref_sct = sctransform(ref_mat, ref_var, params[in(ref_var.id).(params.id),:])

		transformed_proj2 = project(counts_proj2, transformed) # TODO: use me
		# transformed_proj2 = project(counts_proj2, transformed; rtol=1e-9)

		@testset "transform" begin
			@test materialize(transformed_proj2) ≈ ref_sct rtol=1e-3
		end

		@testset "full" begin
			fl_proj2 = project(counts_proj2, fl)
			@test size(fl_proj2) == size(fl)
		end
	end
end
