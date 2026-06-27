using Test
using SingleCellProjections
using SingleCellProjections: SingleCellProjectionsCore
using .SingleCellProjectionsCore: unblockify
using ReproducibleJobs: fetch!, forward!
using Muon: AnnData, writeh5ad
using SparseArrays
using DataFrames
using LinearAlgebra


function create_test_h5ad(path)
	nobs, nvar = 30, 10
	ndim = 3

	X = sprand(Float32, nobs, nvar, 0.3)
	raw_counts = sprand(nobs, nvar, 0.3, k->rand(1:100,k))
	dense = rand(Float32, nobs, nvar)

	obs = DataFrame(cell_type=rand(["A","B","C"], nobs))
	obs_names = ["cell_$i" for i in 1:nobs]

	var = DataFrame(gene_name=["gene_$i" for i in 1:nvar])
	var_names = ["GENE$i" for i in 1:nvar]

	obsm = Dict("X_umap" => rand(Float64, nobs, ndim))
	varm = Dict("PCs" => rand(Float64, nvar, ndim))
	obsp = Dict("distances" => sprand(Float32, nobs, nobs, 0.2))
	varp = Dict("correlations" => rand(Float64, nvar, nvar))

	adata = AnnData(;
		X,
		obs, obs_names, var, var_names,
		obsm, varm, obsp, varp,
		layers=Dict{String,AbstractMatrix{<:Real}}("raw_counts" => Float32.(raw_counts), "dense" => dense),
	)
	writeh5ad(path, adata)

	return (; X, raw_counts, dense, obs, obs_names, var, var_names, obsm, varm, obsp, varp, nobs, nvar, ndim)
end


function run_muon_tests()
	h5ad_path = joinpath(mktempdir(), "test.h5ad")
	ground_truth = create_test_h5ad(h5ad_path)

	@testset "load_h5ad" begin
		@testset "X (default)" begin
			job = SCP.load_h5ad(h5ad_path)
			dm = fetch!(job)

			@test size(dm) == (ground_truth.nvar, ground_truth.nobs)
			@test unblockify(dm.matrix) ≈ ground_truth.X'

			@test dm.var.id == ground_truth.var_names
			@test dm.var.gene_name == ground_truth.var.gene_name
			@test dm.obs.cell_id == ground_truth.obs_names
			@test dm.obs.cell_type == ground_truth.obs.cell_type
		end

		@testset "X with eltype conversion" begin
			job = SCP.load_h5ad(Float64, h5ad_path)
			dm = fetch!(job)

			@test eltype(dm.matrix) == Float64
			@test unblockify(dm.matrix) ≈ Float64.(ground_truth.X')
		end

		@testset "layer (raw_counts)" begin
			job = SCP.load_h5ad(Int, h5ad_path; layer="raw_counts")
			dm = fetch!(job)

			@test size(dm) == (ground_truth.nvar, ground_truth.nobs)
			@test unblockify(dm.matrix) == ground_truth.raw_counts'
			@test eltype(dm.matrix) == Int

			@test dm.var.id == ground_truth.var_names
			@test dm.obs.cell_id == ground_truth.obs_names
		end

		@testset "layer (dense)" begin
			job = SCP.load_h5ad(h5ad_path; layer="dense")
			dm = fetch!(job)

			@test size(dm) == (ground_truth.nvar, ground_truth.nobs)
			@test unblockify(dm.matrix) == ground_truth.dense'
			@test eltype(dm.matrix) == Float32

			@test dm.var.id == ground_truth.var_names
			@test dm.obs.cell_id == ground_truth.obs_names
		end

		@testset "obsm" begin
			job = SCP.load_h5ad(h5ad_path; obsm="X_umap")
			dm = fetch!(job)

			@test size(dm) == (ground_truth.ndim, ground_truth.nobs)
			@test dm.matrix ≈ ground_truth.obsm["X_umap"]'

			@test dm.var.id == ["Dim1", "Dim2", "Dim3"]
			@test dm.obs.cell_id == ground_truth.obs_names
			@test dm.obs.cell_type == ground_truth.obs.cell_type
		end

		@testset "varm" begin
			job = SCP.load_h5ad(h5ad_path; varm="PCs")
			dm = fetch!(job)

			@test size(dm) == (ground_truth.nvar, ground_truth.ndim)
			@test dm.matrix ≈ ground_truth.varm["PCs"]

			@test dm.var.id == ground_truth.var_names
			@test dm.var.gene_name == ground_truth.var.gene_name
			@test dm.obs.id == ["Dim1", "Dim2", "Dim3"]
		end

		@testset "obsp" begin
			job = SCP.load_h5ad(h5ad_path; obsp="distances")
			dm = fetch!(job)

			@test size(dm) == (ground_truth.nobs, ground_truth.nobs)
			@test unblockify(dm.matrix) ≈ ground_truth.obsp["distances"]'

			@test dm.var.cell_id == ground_truth.obs_names
			@test dm.obs.cell_id == ground_truth.obs_names
		end

		@testset "varp" begin
			job = SCP.load_h5ad(h5ad_path; varp="correlations")
			dm = fetch!(job)

			@test size(dm) == (ground_truth.nvar, ground_truth.nvar)
			@test dm.matrix ≈ ground_truth.varp["correlations"]

			@test dm.var.id == ground_truth.var_names
			@test dm.obs.id == ground_truth.var_names
		end

		@testset "mutually exclusive kwargs" begin
			@test_throws ArgumentError SCP.load_h5ad(h5ad_path; layer="raw_counts", obsm="X_umap")
		end

		@testset "var/obs sharing across sources" begin
			job_x = SCP.load_h5ad(h5ad_path)
			var = forward!(SCP.get_var(job_x))
			obs = forward!(SCP.get_obs(job_x))

			job_raw = SCP.load_h5ad(h5ad_path; layer="raw_counts")
			@test forward!(SCP.get_var(job_raw)) === var
			@test forward!(SCP.get_obs(job_raw)) === obs

			job_obsp = SCP.load_h5ad(h5ad_path; obsp="distances")
			@test forward!(SCP.get_var(job_obsp)) === obs
			@test forward!(SCP.get_obs(job_obsp)) === obs

			job_obsm = SCP.load_h5ad(h5ad_path; obsm="X_umap")
			@test forward!(SCP.get_obs(job_obsm)) === obs

			job_varp = SCP.load_h5ad(h5ad_path; varp="correlations")
			@test forward!(SCP.get_var(job_varp)) === var
			@test forward!(SCP.get_obs(job_varp)) === var

			job_varm = SCP.load_h5ad(h5ad_path; varm="PCs")
			@test forward!(SCP.get_var(job_varm)) === var
		end
	end
end
