isgz(fn) = lowercase(splitext(fn)[2])==".gz"
_open(f, fn) = open(fn) do io
    f(isgz(fn) ? GzipDecompressorStream(io) : io)
end

read_matrix(fn,delim=',') = _open(io->readdlm(io,delim,Int), fn)
read_strings(fn,delim=',') = _open(io->readdlm(io,delim,String), fn)

function simple_logtransform(X, scale_factor)
	s = sum(X; dims=1)
	log2.( 1 .+ X.*scale_factor./max.(1,s) )
end

materialize(X) = X
materialize(X::MatrixExpression) = X*I(size(X,2))
materialize(X::SVD) = convert(Matrix,X)
materialize(X::SingleCellProjections.LowRank) = X.U*X.Vt
materialize(X::DataMatrix) = materialize(X.matrix)

function pairwise_dist(X)
	K = X'X
	d = diag(K)
	D2 = d .+ d' .- 2.0.*K
	sqrt.(max.(0.0, D2))
end

function ncommon_neighbors(A,B; k=20)
	@assert size(A,2)==size(B,2)
	N = size(A,2)
	Dr = pairwise_dist(A)
	Df = pairwise_dist(B)
	ncommon = zeros(Int,N)
	for i in 1:N
		a = sortperm(Dr[:,i])[1:k]
		b = sortperm(Df[:,i])[1:k]
		ncommon[i] = length(intersect(a,b))
	end
	ncommon
end


@testset "Basic Workflow" begin
	pbmc_path = joinpath(pkgdir(SingleCellProjections), "test/data/500_PBMC_3p_LT_Chromium_X_50genes")
	h5_path = joinpath(pbmc_path, "filtered_feature_bc_matrix.h5")
	mtx_path = joinpath(pbmc_path, "filtered_feature_bc_matrix/matrix.mtx.gz")

    P,N = (50,587)

    expected_mat = read_matrix(joinpath(pbmc_path,"expected_matrix.csv"))
    expected_nnz = count(!iszero, expected_mat)
    expected_feature_ids = vec(read_strings(joinpath(pbmc_path,"expected_feature_ids.csv")))
    expected_barcodes = vec(read_strings(joinpath(pbmc_path,"expected_barcodes.csv")))

    expected_feature_names = read_strings(joinpath(pbmc_path,"filtered_feature_bc_matrix/features.tsv.gz"),'\t')[:,2]
    expected_feature_types = fill("Gene Expression", P)
    expected_feature_genome = fill("GRCh38", P)

	@testset "load10x $(split(basename(p),'.';limit=2)[2]) lazy=$lazy" for p in (h5_path,mtx_path), lazy in (false, true)
		counts = load10x(p; lazy)
		@test size(counts)==(P,N)
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

		@test size(counts)==(P,N*2)
		@test nnz(counts.matrix) == expected_nnz*2

		@test Set(names(counts.obs)) == Set(("id", "sampleName", "barcode"))
		@test counts.obs.id == [string.("a_",expected_barcodes); string.("b_",expected_barcodes)]
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

	counts.obs.group = rand(StableRNG(904), ("A","B","C"), size(counts,2))
	counts.obs.value = 1 .+ randn(StableRNG(905), size(counts,2))

	# dataset for projection - by using a subset of the obs in `counts`, we make unit testing simpler while still testing well
	counts_proj = filter_obs(row->row.group!="B" && row.value>0.6, counts)
	empty!(counts_proj.models)

	proj_obs_indices = identity.(indexin(counts_proj.obs.barcode, counts.obs.barcode))


	@testset "logtransform scale_factor=$scale_factor" for scale_factor in (10_000, 1_000)
		X = simple_logtransform(expected_mat, scale_factor)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)
		l = logtransform(counts; kwargs...)
		@test l.matrix.matrix ≈ X
		@test nnz(l.matrix.matrix) == expected_nnz

		lproj = project(counts_proj, l)
		@test lproj.matrix.matrix ≈ X[:,proj_obs_indices]
	end

	transformed = sctransform(counts; use_cache=false)
	transformed_proj = project(counts_proj, transformed)
	@testset "sctransform" begin
		params = scparams(counts.matrix, counts.var; use_cache=false)

		@test params.logGeneMean ≈ transformed.var.logGeneMean
		@test params.outlier == transformed.var.outlier
		@test params.beta0 ≈ transformed.var.beta0
		@test params.beta1 ≈ transformed.var.beta1
		@test params.theta ≈ transformed.var.theta

		sct = sctransform(counts.matrix, counts.var, params)

		@test size(transformed.matrix) == size(sct)
		@test materialize(transformed) ≈ sct rtol=1e-3

		@test materialize(transformed_proj) ≈ sct[:,proj_obs_indices] rtol=1e-3

		@test params.logGeneMean ≈ transformed_proj.var.logGeneMean
		@test params.outlier == transformed_proj.var.outlier
		@test params.beta0 ≈ transformed_proj.var.beta0
		@test params.beta1 ≈ transformed_proj.var.beta1
		@test params.theta ≈ transformed_proj.var.theta
	end

	# TODO: tf_idf_transform

	X = materialize(transformed)
	Xc = (X.-mean(X; dims=2))
	Xs = Xc ./ std(X; dims=2)

	# categorical
	Xcat = copy(X)
	g = transformed.obs.group
	for c in unique(g)
		Xcat[:, c.==g] .-= mean(Xcat[:, c.==g]; dims=2)
	end
	Xcat_s = Xcat ./ std(Xcat; dims=2)

	# numerical
	v = transformed.obs.value .- mean(transformed.obs.value)
	β = Xc/v'
	Xnum = Xc .- β*v'
	Xnum_s = Xnum ./ std(Xnum; dims=2)

	# combined
	D = [g.=="A" g.=="B" g.=="C" v]
	β = X / D'
	Xcom = X .- β*D'
	Xcom_s = Xcom ./ std(Xcom; dims=2)

	@testset "normalize" begin
		normalized = normalize_matrix(transformed)
		@test materialize(normalized) ≈ Xc
		@test materialize(project(counts_proj,normalized)) ≈ Xc[:,proj_obs_indices] rtol=1e-3
		@test materialize(project(transformed_proj,normalized)) ≈ Xc[:,proj_obs_indices] rtol=1e-3
		normalized = normalize_matrix(transformed; scale=true)
		@test materialize(normalized) ≈ Xs
		@test materialize(project(transformed_proj,normalized)) ≈ Xs[:,proj_obs_indices] rtol=1e-3

		normalized = normalize_matrix(transformed, "group")
		@test materialize(normalized) ≈ Xcat
		@test materialize(project(transformed_proj,normalized)) ≈ Xcat[:,proj_obs_indices] rtol=1e-3
		normalized = normalize_matrix(transformed, "group"; scale=true)
		@test materialize(normalized) ≈ Xcat_s
		@test materialize(project(transformed_proj,normalized)) ≈ Xcat_s[:,proj_obs_indices] rtol=1e-3

		normalized = normalize_matrix(transformed, "value")
		@test materialize(normalized) ≈ Xnum
		@test materialize(project(transformed_proj,normalized)) ≈ Xnum[:,proj_obs_indices] rtol=1e-3
		normalized = normalize_matrix(transformed, "value"; scale=true)
		@test materialize(normalized) ≈ Xnum_s
		@test materialize(project(transformed_proj,normalized)) ≈ Xnum_s[:,proj_obs_indices] rtol=1e-3

		normalized = normalize_matrix(transformed, "group", "value")
		@test materialize(normalized) ≈ Xcom
		@test materialize(project(transformed_proj,normalized)) ≈ Xcom[:,proj_obs_indices] rtol=1e-3
		normalized = normalize_matrix(transformed, "group", "value"; scale=true)
		@test materialize(normalized) ≈ Xcom_s
		@test materialize(project(transformed_proj,normalized)) ≈ Xcom_s[:,proj_obs_indices] rtol=1e-3

	end

	normalized = normalize_matrix(transformed, "group", "value")
	normalized_proj = project(transformed_proj,normalized)

	@testset "svd" begin
		reduced = svd(normalized; nsv=3, subspacedims=24, niter=4, rng=StableRNG(102))
		F = svd(Xcom)
		@test size(reduced)==size(transformed)
		@test reduced.matrix.S ≈ F.S[1:3] rtol=1e-3
		@test abs.(reduced.matrix.U'F.U[:,1:3]) ≈ I(3) rtol=1e-3
		@test abs.(reduced.matrix.V'F.V[:,1:3]) ≈ I(3) rtol=1e-3

		U = reduced.matrix.U
		@test all(>(0.0), sum(U;dims=1))

		X = materialize(reduced)
		reduced_proj = project(normalized_proj, reduced)
		Xproj = materialize(reduced_proj)
		@test Xproj ≈ X[:,proj_obs_indices] rtol=1e-3

		U_proj = reduced_proj.matrix.U
		@test all(>(0.0), sum(U_proj;dims=1))
	end

	reduced = svd(normalized; nsv=10, niter=4, rng=StableRNG(102))
	reduced_proj = project(normalized_proj, reduced)

	@testset "filter $name" for (name,data,data_proj) in (("counts",counts,counts_proj), ("normalized",normalized,normalized_proj), ("reduced",reduced,reduced_proj))
		P2 = size(data,1)
		X = materialize(data)

		f = filter_var(1:2:P2, data)
		@test materialize(f) ≈ X[1:2:P2, :]
		@test f.obs == data.obs
		@test f.var == data.var[1:2:P2, :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[1:2:P2, proj_obs_indices] rtol=1e-3
		@test f_proj.obs == data.obs[proj_obs_indices, :]
		@test f_proj.var == data.var[1:2:P2, :]

		f = data[1:2:end,:]
		@test materialize(f) ≈ X[1:2:end, :]
		@test f.obs == data.obs
		@test f.var == data.var[1:2:end, :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[1:2:end, proj_obs_indices] rtol=1e-3
		@test f_proj.obs == data.obs[proj_obs_indices, :]
		@test f_proj.var == data.var[1:2:end, :]

		f = filter_obs(1:10:N, data)
		@test materialize(f) ≈ X[:, 1:10:N]
		@test f.obs == data.obs[1:10:N, :]
		@test f.var == data.var
		@test_throws ArgumentError project(data_proj, f)

		f = data[:,1:10:end]
		@test materialize(f) ≈ X[:, 1:10:end]
		@test f.obs == data.obs[1:10:end, :]
		@test f.var == data.var
		@test_throws ArgumentError project(data_proj, f)

		f = filter_matrix(1:2:P2, 1:10:N, data)
		@test materialize(f) ≈ X[1:2:P2, 1:10:N]
		@test f.obs == data.obs[1:10:N, :]
		@test f.var == data.var[1:2:P2, :]
		@test_throws ArgumentError project(data_proj, f)

		f = data[1:2:end,1:10:end]
		@test materialize(f) ≈ X[1:2:end, 1:10:end]
		@test f.obs == data.obs[1:10:end, :]
		@test f.var == data.var[1:2:end, :]
		@test_throws ArgumentError project(data_proj, f)

		f = filter_obs("group"=>==("A"), data)
		@test materialize(f) ≈ X[:, g.=="A"]
		@test f.obs == data.obs[g.=="A", :]
		@test f.var == data.var
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[:, obs_ind] rtol=1e-3
		@test f_proj.obs == data.obs[obs_ind, :]
		@test f_proj.var == data.var

		f = filter_obs(row->row.group=="A", data)
		@test materialize(f) ≈ X[:, g.=="A"]
		@test f.obs == data.obs[g.=="A", :]
		@test f.var == data.var
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[:, obs_ind] rtol=1e-3
		@test f_proj.obs == data.obs[obs_ind, :]
		@test f_proj.var == data.var

		f = filter_matrix(1:2:P2, "group"=>==("A"), data)
		@test materialize(f) ≈ X[1:2:P2, g.=="A"]
		@test f.obs == data.obs[g.=="A", :]
		@test f.var == data.var[1:2:P2, :]
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[1:2:P2, obs_ind] rtol=1e-3
		@test f_proj.obs == data.obs[obs_ind, :]
		@test f_proj.var == data.var[1:2:P2, :]

		f = filter_var("name"=>>("D"), data)
		@test materialize(f) ≈ X[data.var.name.>="D", :]
		@test f.obs == data.obs
		@test f.var == data.var[data.var.name.>="D", :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[data.var.name.>="D", proj_obs_indices] rtol=1e-3
		@test f_proj.obs == data.obs[proj_obs_indices, :]
		@test f_proj.var == data.var[data.var.name.>="D", :]

		f = filter_var(row->row.name>"D", data)
		@test materialize(f) ≈ X[data.var.name.>="D", :]
		@test f.obs == data.obs
		@test f.var == data.var[data.var.name.>="D", :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[data.var.name.>="D", proj_obs_indices] rtol=1e-3
		@test f_proj.obs == data.obs[proj_obs_indices, :]
		@test f_proj.var == data.var[data.var.name.>="D", :]

		f = filter_matrix("name"=>>("D"), 1:10:N, data)
		@test materialize(f) ≈ X[data.var.name.>="D", 1:10:N]
		@test f.obs == data.obs[1:10:N, :]
		@test f.var == data.var[data.var.name.>="D", :]
		@test_throws ArgumentError project(data_proj, f)

		f = filter_matrix("name"=>>("D"), "group"=>==("A"), data)
		@test materialize(f) ≈ X[data.var.name.>="D", g.=="A"]
		@test f.obs == data.obs[g.=="A", :]
		@test f.var == data.var[data.var.name.>="D", :]
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[data.var.name.>="D", obs_ind] rtol=1e-3
		@test f_proj.obs == data.obs[obs_ind, :]
		@test f_proj.var == data.var[data.var.name.>="D", :]
	end


	@testset "force layout seed=$seed" for seed in 1:5
		fl = force_layout(reduced; ndim=3, k=10, rng=StableRNG(seed))
		# Sanity check output by checking that there is a descent overlap between nearest neighbors
		ncommon = ncommon_neighbors(obs_coordinates(fl), obs_coordinates(reduced))
		@test mean(ncommon) > 8

		fl_proj = project(reduced_proj, fl)
		@test materialize(fl_proj)≈materialize(fl)[:,proj_obs_indices] rtol=1e-5
	end

	@testset "UMAP" begin
		umapped = umap(reduced, 3)
		# Sanity check output by checking that there is a descent overlap between nearest neighbors
		ncommon = ncommon_neighbors(obs_coordinates(umapped), obs_coordinates(reduced))
		@test mean(ncommon) > 9

		# TODO: can we check this in a better way? Projection into a UMAP is not very exact.
		umapped_proj = project(reduced_proj, umapped)
		@test materialize(umapped_proj)≈materialize(umapped)[:,proj_obs_indices] rtol=1e-1
	end

	@testset "t-SNE" begin
		t = tsne(reduced, 3)
		# Sanity check output by checking that there is a descent overlap between nearest neighbors
		ncommon = ncommon_neighbors(obs_coordinates(t), obs_coordinates(reduced))
		@test mean(ncommon) > 9

		t_proj = project(reduced_proj, t)
		@test materialize(t_proj)≈materialize(t)[:,proj_obs_indices] rtol=1e-5
	end

	@testset "var_counts_fraction" begin
		X = counts.matrix
		nA = vec(sum(X[startswith.(counts.var.name,"A"),:]; dims=1))
		nAC = vec(sum(X[startswith.(counts.var.name,"AC"),:]; dims=1))
		nTot = vec(sum(X;dims=1))

		c = DataMatrix(counts.matrix, counts.var, counts.obs) # TODO: copy(counts)

		var_counts_fraction!(c, "name"=>startswith("A"), Returns(true), "A")
		@test c.obs.A == nA ./ nTot

		var_counts_fraction!(c, "name"=>startswith("AC"), "name"=>startswith("A"), "B")
		@test c.obs.B == nAC ./ max.(1,nA)

		@test_throws ArgumentError var_counts_fraction!(c, "name"=>startswith("NOTAGENE"), Returns(true), "C")
		@test "C" ∉ names(c.obs)

		var_counts_fraction!(c, "name"=>startswith("NOTAGENE"), Returns(true), "C"; check=false)
		@test all(iszero, c.obs.C)

		c_proj = project(counts_proj, c)
		@test c_proj.obs.A == c.obs.A[proj_obs_indices]
		@test c_proj.obs.B == c.obs.B[proj_obs_indices]
		@test c_proj.obs.C == c.obs.C[proj_obs_indices]
	end

	# TODO: var2obs
	# TODO: projections
end
