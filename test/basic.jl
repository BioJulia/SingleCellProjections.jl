@testset "Basic Workflow" begin
	P,N = (50,587)

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

		test_show(l; matrix="SparseMatrixCSC", var=names(counts.var), obs=names(counts.obs), models="LogTransformModel")
		test_show(lproj; matrix="SparseMatrixCSC", var=names(counts_proj.var), obs=names(counts_proj.obs), models="LogTransformModel")
	end

	@testset "tf-idf scale_factor=$scale_factor" for scale_factor in (10_000, 1_000)
		idf = simple_idf(expected_mat)
		X = simple_tf_idf_transform(expected_mat, idf, scale_factor)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)
		tf = tf_idf_transform(counts; kwargs...)
		@test tf.matrix.matrix ≈ X
		@test nnz(tf.matrix.matrix) == expected_nnz

		tf_proj = project(counts_proj, tf)
		@test tf_proj.matrix.matrix ≈ X[:,proj_obs_indices]

		test_show(tf; matrix="SparseMatrixCSC", var=vcat(names(counts.var),"idf"), obs=names(counts.obs), models="TFIDFTransformModel")
		test_show(tf_proj; matrix="SparseMatrixCSC", var=vcat(names(counts_proj.var),"idf"), obs=names(counts_proj.obs), models="TFIDFTransformModel")
	end

	transformed_proj = project(counts_proj, transformed)
	@testset "sctransform" begin
		@test params.logGeneMean ≈ transformed.var.logGeneMean
		@test params.outlier == transformed.var.outlier
		@test params.beta0 ≈ transformed.var.beta0
		@test params.beta1 ≈ transformed.var.beta1
		@test params.theta ≈ transformed.var.theta

		sct = sctransform(expected_sparse, counts.var, params)

		@test size(transformed.matrix) == size(sct)
		@test materialize(transformed) ≈ sct rtol=1e-3

		@test materialize(transformed_proj) ≈ sct[:,proj_obs_indices] rtol=1e-3

		@test params.logGeneMean ≈ transformed_proj.var.logGeneMean
		@test params.outlier == transformed_proj.var.outlier
		@test params.beta0 ≈ transformed_proj.var.beta0
		@test params.beta1 ≈ transformed_proj.var.beta1
		@test params.theta ≈ transformed_proj.var.theta

		test_show(transformed; matrix=r"^A\+B₁B₂B₃$", models="SCTransformModel")

		t2 = sctransform(counts; use_cache=false, var_filter=nothing)
		@test materialize(t2) ≈ sct rtol=1e-3
	end

	X = materialize(transformed)
	Xc = (X.-mean(X; dims=2))
	X_std = std(X; dims=2)
	Xs = Xc ./ X_std

	# categorical
	Xcat = copy(X)
	g = transformed.obs.group
	for c in unique(g)
		Xcat[:, c.==g] .-= mean(Xcat[:, c.==g]; dims=2)
	end
	Xcat_std = std(Xcat; dims=2)
	Xcat_s = Xcat ./ Xcat_std

	# numerical
	v = transformed.obs.value .- mean(transformed.obs.value)
	β = Xc/v'
	Xnum = Xc .- β*v'
	Xnum_std = std(Xnum; dims=2)
	Xnum_s = Xnum ./ Xnum_std

	# combined
	D = [g.=="A" g.=="B" g.=="C" v]
	β = X / D'
	Xcom = X .- β*D'
	Xcom_std = std(Xcom; dims=2)
	Xcom_s = Xcom ./ Xcom_std

	# two-group
	D = [g.=="C" g.!="C"]
	β = X / D'
	Xtwo = X .- β*D'
	Xtwo_std = std(Xtwo; dims=2)
	Xtwo_s = Xtwo ./ Xtwo_std

	@testset "normalize" begin
		n = normalize_matrix(transformed)
		@test materialize(n) ≈ Xc
		@test materialize(project(counts_proj,n)) ≈ Xc[:,proj_obs_indices] rtol=1e-3
		@test materialize(project(transformed_proj,n)) ≈ Xc[:,proj_obs_indices] rtol=1e-3
		test_show(n; matrix=r"^A\+B₁B₂B₃\+\(-β\)X'$", models="NormalizationModel")
		n = normalize_matrix(transformed; scale=true)
		@test materialize(n) ≈ Xs
		@test materialize(project(transformed_proj,n)) ≈ Xs[:,proj_obs_indices] rtol=1e-3
		@test n.var.scaling ≈ 1.0./X_std
		test_show(n; matrix=r"^D\(A\+B₁B₂B₃\+\(-β\)X'\)$", models="NormalizationModel")

		n = normalize_matrix(transformed, "group")
		@test materialize(n) ≈ Xcat
		@test materialize(project(transformed_proj,n)) ≈ Xcat[:,proj_obs_indices] rtol=1e-3
		n = normalize_matrix(transformed, "group"; scale=true)
		@test materialize(n) ≈ Xcat_s
		@test materialize(project(transformed_proj,n)) ≈ Xcat_s[:,proj_obs_indices] rtol=1e-3
		@test n.var.scaling ≈ 1.0./Xcat_std

		n = normalize_matrix(transformed, "value")
		@test materialize(n) ≈ Xnum
		@test materialize(project(transformed_proj,n)) ≈ Xnum[:,proj_obs_indices] rtol=1e-3
		n = normalize_matrix(transformed, "value"; scale=true)
		@test materialize(n) ≈ Xnum_s
		@test materialize(project(transformed_proj,n)) ≈ Xnum_s[:,proj_obs_indices] rtol=1e-3
		@test n.var.scaling ≈ 1.0./Xnum_std

		n = normalize_matrix(transformed, "group", "value")
		@test materialize(n) ≈ Xcom
		@test materialize(project(transformed_proj,n)) ≈ Xcom[:,proj_obs_indices] rtol=1e-3
		n = normalize_matrix(transformed, "group", "value"; scale=true)
		@test materialize(n) ≈ Xcom_s
		@test materialize(project(transformed_proj,n)) ≈ Xcom_s[:,proj_obs_indices] rtol=1e-3
		@test n.var.scaling ≈ 1.0./Xcom_std

		n = normalize_matrix(transformed, covariate("group","C"))
		@test materialize(n) ≈ Xtwo
		@test materialize(project(transformed_proj,n)) ≈ Xtwo[:,proj_obs_indices] rtol=1e-3
		n = normalize_matrix(transformed, covariate("group","C"); scale=true)
		@test materialize(n) ≈ Xtwo_s
		@test materialize(project(transformed_proj,n)) ≈ Xtwo_s[:,proj_obs_indices] rtol=1e-3
		@test n.var.scaling ≈ 1.0./Xtwo_std

		n = normalize_matrix(transformed, covariate("group","B"))
		@test_throws "No values" project(transformed_proj,n)
	end

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

		@test var_coordinates(reduced) == reduced.matrix.U

		X = materialize(reduced)
		reduced_proj = project(normalized_proj, reduced)
		Xproj = materialize(reduced_proj)
		@test Xproj ≈ X[:,proj_obs_indices] rtol=1e-3

		U_proj = reduced_proj.matrix.U
		@test all(>(0.0), sum(U_proj;dims=1))

		test_show(reduced; matrix="SVD (3 dimensions)", models="SVDModel")
	end

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

		test_show(f, models="FilterModel")

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

		test_show(fl; matrix="Matrix{Float64}", models="NearestNeighborModel")
	end

	@testset "UMAP" begin
		umapped = umap(reduced, 3)
		# Sanity check output by checking that there is a descent overlap between nearest neighbors
		ncommon = ncommon_neighbors(obs_coordinates(umapped), obs_coordinates(reduced))
		@test mean(ncommon) > 9

		# TODO: can we check this in a better way? Projection into a UMAP is not very exact.
		umapped_proj = project(reduced_proj, umapped)
		# Hmm. this fails sometimes since it is non-deterministic.
		# @test materialize(umapped_proj)≈materialize(umapped)[:,proj_obs_indices] rtol=1e-1
		@test size(umapped_proj) == (size(umapped,1),length(proj_obs_indices))

		test_show(umapped; matrix="Matrix{Float64}", models="UMAPModel")
	end

	@testset "t-SNE" begin
		t = tsne(reduced, 3)
		# Sanity check output by checking that there is a descent overlap between nearest neighbors
		ncommon = ncommon_neighbors(obs_coordinates(t), obs_coordinates(reduced))
		@test mean(ncommon) > 9

		t_proj = project(reduced_proj, t)
		@test materialize(t_proj)≈materialize(t)[:,proj_obs_indices] rtol=1e-5

		test_show(t; matrix="Matrix{Float64}", models="NearestNeighborModel")
	end

	@testset "var_counts_fraction" begin
		X = counts.matrix
		nA = vec(sum(X[startswith.(counts.var.name,"A"),:]; dims=1))
		nAC = vec(sum(X[startswith.(counts.var.name,"AC"),:]; dims=1))
		nTot = vec(sum(X;dims=1))

		c = copy(counts)

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

		test_show(c; obs=vcat(names(counts.obs), ["A","B","C"]), models="VarCountsFractionModel")
	end
end
