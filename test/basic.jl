add_id_prefix!(df::DataFrame, prefix) = (df[:,1] = string.(prefix, df[:,1]); df)
add_id_prefix(df::DataFrame, prefix) = add_id_prefix!(copy(df; copycols=false), prefix)

@testset "Basic Workflow" begin
	P,N = (50,587)

	# dataset for projection - by using a subset of the obs in `counts`, we make unit testing simpler while still testing well. But rename obs IDs to ensure they are treated as separate obs.
	counts_proj = filter_obs(row->row.group!="B" && row.value>0.6, counts)
	empty!(counts_proj.models)
	proj_obs_indices = identity.(indexin(counts_proj.obs.barcode, counts.obs.barcode))
	add_id_prefix!(counts_proj.obs, "proj_")


	@testset "logtransform scale_factor=$scale_factor T=$T" for scale_factor in (10_000, 1_000), T in (Float64,Float32)
		X = simple_logtransform(expected_mat, scale_factor)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)
		if T==Float64
			l = logtransform(counts; kwargs...)
		else
			l = logtransform(T, counts; kwargs...)
			X = T.(X)
		end

		@test l.matrix.matrix ≈ X
		@test nnz(l.matrix.matrix) == expected_nnz
		@test eltype(l.matrix.matrix) == T

		lproj = project(counts_proj, l)
		@test lproj.matrix.matrix ≈ X[:,proj_obs_indices]
		@test eltype(lproj.matrix.matrix) == T

		test_show(l; matrix="SparseMatrixCSC", var=names(counts.var), obs=names(counts.obs), models="LogTransformModel")
		test_show(lproj; matrix="SparseMatrixCSC", var=names(counts_proj.var), obs=names(counts_proj.obs), models="LogTransformModel")
	end

	@testset "tf-idf scale_factor=$scale_factor T=$T" for scale_factor in (10_000, 1_000), T in (Float64,Float32)
		idf = simple_idf(expected_mat)
		X = simple_tf_idf_transform(expected_mat, idf, scale_factor)
		kwargs = scale_factor == 10_000 ? (;) : (;scale_factor)

		if T==Float64
			tf = tf_idf_transform(counts; kwargs...)
		else
			tf = tf_idf_transform(T, counts; kwargs...)
			X = T.(X)
		end

		@test tf.matrix.matrix ≈ X
		@test nnz(tf.matrix.matrix) == expected_nnz
		@test eltype(tf.matrix.matrix) == T

		tf_proj = project(counts_proj, tf)
		@test tf_proj.matrix.matrix ≈ X[:,proj_obs_indices]
		@test eltype(tf_proj.matrix.matrix) == T

		test_show(tf; matrix="SparseMatrixCSC", var=vcat(names(counts.var),"idf"), obs=names(counts.obs), models="TFIDFTransformModel")
		test_show(tf_proj; matrix="SparseMatrixCSC", var=vcat(names(counts_proj.var),"idf"), obs=names(counts_proj.obs), models="TFIDFTransformModel")
	end

	transformed_proj = project(counts_proj, transformed)
	@testset "sctransform T=$T" for T in (Float64,Float32)
		sct = sctransform(expected_sparse, counts.var, params)

		if T==Float64
			trans = transformed
			trans_proj = transformed_proj
			t2 = sctransform(counts; use_cache=false, var_filter=nothing)
		else
			trans = sctransform(Float32, counts; use_cache=false)
			trans_proj = project(counts_proj, trans)
			t2 = sctransform(T, counts; use_cache=false, var_filter=nothing)
		end

		@test params.logGeneMean ≈ trans.var.logGeneMean
		@test params.outlier == trans.var.outlier
		@test params.beta0 ≈ trans.var.beta0
		@test params.beta1 ≈ trans.var.beta1
		@test params.theta ≈ trans.var.theta

		@test size(trans.matrix) == size(sct)
		@test eltype(trans.matrix.terms[1].matrix) == T
		@test materialize(trans) ≈ sct rtol=1e-3

		@test eltype(trans_proj.matrix.terms[1].matrix) == T
		@test materialize(trans_proj) ≈ sct[:,proj_obs_indices] rtol=1e-3

		@test params.logGeneMean ≈ trans_proj.var.logGeneMean
		@test params.outlier == trans_proj.var.outlier
		@test params.beta0 ≈ trans_proj.var.beta0
		@test params.beta1 ≈ trans_proj.var.beta1
		@test params.theta ≈ trans_proj.var.theta

		test_show(trans; matrix=r"^A\+B₁B₂B₃$", models="SCTransformModel")

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

	obs2_df = rename(counts.obs, "group"=>"external_group", "value"=>"external_value")
	obs2_proj_df = rename(counts_proj.obs, "group"=>"external_group", "value"=>"external_value")

	@testset "normalize (external_obs::$T)" for T in (Annotations,DataFrame)
		if T==DataFrame
			obs2 = obs2_df
			obs2_proj = obs2_proj_df
			obs2_eg = select(obs2, ["barcode","external_group"])
			obs2_ev = select(obs2, ["barcode","external_value"])
			obs2_proj_eg = select(obs2_proj, ["barcode","external_group"])
			obs2_proj_ev = select(obs2_proj, ["barcode","external_value"])
		else
			obs2 = T(obs2_df)
			obs2_proj = T(obs2_proj_df)
			obs2_eg = obs2.external_group
			obs2_ev = obs2.external_value
			obs2_proj_eg = obs2_proj.external_group
			obs2_proj_ev = obs2_proj.external_value
		end

		@test_throws ["ArgumentError", "exactly one ID column", "one value column"] normalize_matrix(transformed, obs2)

		n = normalize_matrix(transformed, obs2_eg)
		@test materialize(n) ≈ Xcat
		@test_throws ["ArgumentError", "external_group"] project(transformed_proj,n)
		@test materialize(project(transformed_proj,n; external_obs=obs2_proj_eg)) ≈ Xcat[:,proj_obs_indices] rtol=1e-3
		@test materialize(project(transformed_proj,n; external_obs=obs2_proj)) ≈ Xcat[:,proj_obs_indices] rtol=1e-3

		n = normalize_matrix(transformed, obs2_ev)
		@test materialize(n) ≈ Xnum
		@test_throws ["ArgumentError", "external_value"] project(transformed_proj,n)
		@test materialize(project(transformed_proj,n; external_obs=obs2_proj_ev)) ≈ Xnum[:,proj_obs_indices] rtol=1e-3
		@test materialize(project(transformed_proj,n; external_obs=obs2_proj)) ≈ Xnum[:,proj_obs_indices] rtol=1e-3

		n = normalize_matrix(transformed, covariate(obs2_eg,"C"))
		@test materialize(n) ≈ Xtwo
		@test_throws ["ArgumentError", "external_group"] project(transformed_proj,n)
		@test materialize(project(transformed_proj,n; external_obs=obs2_proj_eg)) ≈ Xtwo[:,proj_obs_indices] rtol=1e-3
		@test materialize(project(transformed_proj,n; external_obs=obs2_proj)) ≈ Xtwo[:,proj_obs_indices] rtol=1e-3

		n = normalize_matrix(transformed, covariate(obs2_eg,"B"))
		@test_throws ["ArgumentError", "external_group"] project(transformed_proj,n)
		@test_throws "No values" project(transformed_proj,n; external_obs=obs2_proj_eg)
		@test_throws "No values" project(transformed_proj,n; external_obs=obs2_proj)
	end


	normalized_proj = project(transformed_proj,normalized)

	@testset "svd" begin
		reduced = svd(normalized; nsv=3, subspacedims=24, niter=4, seed=102)
		F = svd(Xcom)
		@test size(reduced)==size(normalized)
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

	@testset "PrincipalMomentAnalysis" begin
		G = groupsimplices(normalized.obs.group)
		p = pma(normalized, G; nsv=3, subspacedims=24, niter=8, rng=StableRNG(102))

		F = pma(Xcom, G; nsv=3)
		@test size(p)==size(normalized)
		@test p.matrix.S ≈ F.S rtol=1e-3

		@test abs.(p.matrix.U'F.U) ≈ I(3) rtol=1e-3

		signs = 2 .* (diag(p.matrix.U'F.U) .> 0) .- 1
		FV = F.V*Diagonal(signs)
		@test p.matrix.V ≈ FV rtol=1e-3

		U = p.matrix.U
		@test all(>(0.0), sum(U;dims=1))

		@test var_coordinates(p) == p.matrix.U

		X = materialize(p)
		p_proj = project(normalized_proj, p)
		Xproj = materialize(p_proj)
		@test Xproj ≈ X[:,proj_obs_indices] rtol=1e-3

		U_proj = p_proj.matrix.U
		@test all(>(0.0), sum(U_proj;dims=1))

		test_show(p; matrix="PMA (3 dimensions)", models="PMAModel")
	end


	reduced_proj = project(normalized_proj, reduced)

	@testset "filter $name" for (name,data,data_proj) in (("counts",counts,counts_proj), ("normalized",normalized,normalized_proj), ("reduced",reduced,reduced_proj))
		P2 = size(data,1)
		X = materialize(data)

		obs_ans = add_id_prefix(data.obs, "proj_")

		f = filter_var(1:2:P2, data)
		@test materialize(f) ≈ X[1:2:P2, :]
		@test f.obs == obs_ans
		@test f.var == data.var[1:2:P2, :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[1:2:P2, proj_obs_indices] rtol=1e-3
		@test f_proj.obs == obs_ans[proj_obs_indices, :]
		@test f_proj.var == data.var[1:2:P2, :]

		test_show(f, models="FilterModel")

		f = data[1:2:end,:]
		@test materialize(f) ≈ X[1:2:end, :]
		@test f.obs == obs_ans
		@test f.var == data.var[1:2:end, :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[1:2:end, proj_obs_indices] rtol=1e-3
		@test f_proj.obs == obs_ans[proj_obs_indices, :]
		@test f_proj.var == data.var[1:2:end, :]

		f = filter_obs(1:10:N, data)
		@test materialize(f) ≈ X[:, 1:10:N]
		@test f.obs == obs_ans[1:10:N, :]
		@test f.var == data.var
		@test_throws ArgumentError project(data_proj, f)

		f = data[:,1:10:end]
		@test materialize(f) ≈ X[:, 1:10:end]
		@test f.obs == obs_ans[1:10:end, :]
		@test f.var == data.var
		@test_throws ArgumentError project(data_proj, f)

		f = filter_matrix(1:2:P2, 1:10:N, data)
		@test materialize(f) ≈ X[1:2:P2, 1:10:N]
		@test f.obs == obs_ans[1:10:N, :]
		@test f.var == data.var[1:2:P2, :]
		@test_throws ArgumentError project(data_proj, f)

		f = data[1:2:end,1:10:end]
		@test materialize(f) ≈ X[1:2:end, 1:10:end]
		@test f.obs == obs_ans[1:10:end, :]
		@test f.var == data.var[1:2:end, :]
		@test_throws ArgumentError project(data_proj, f)

		f = filter_obs("group"=>==("A"), data)
		@test materialize(f) ≈ X[:, g.=="A"]
		@test f.obs == obs_ans[g.=="A", :]
		@test f.var == data.var
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[:, obs_ind] rtol=1e-3
		@test f_proj.obs == obs_ans[obs_ind, :]
		@test f_proj.var == data.var

		f = filter_obs(row->row.group=="A", data)
		@test materialize(f) ≈ X[:, g.=="A"]
		@test f.obs == obs_ans[g.=="A", :]
		@test f.var == data.var
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[:, obs_ind] rtol=1e-3
		@test f_proj.obs == obs_ans[obs_ind, :]
		@test f_proj.var == data.var

		f = filter_matrix(1:2:P2, "group"=>==("A"), data)
		@test materialize(f) ≈ X[1:2:P2, g.=="A"]
		@test f.obs == obs_ans[g.=="A", :]
		@test f.var == data.var[1:2:P2, :]
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[1:2:P2, obs_ind] rtol=1e-3
		@test f_proj.obs == obs_ans[obs_ind, :]
		@test f_proj.var == data.var[1:2:P2, :]

		f = filter_var("name"=>>("D"), data)
		@test materialize(f) ≈ X[data.var.name.>="D", :]
		@test f.obs == obs_ans
		@test f.var == data.var[data.var.name.>="D", :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[data.var.name.>="D", proj_obs_indices] rtol=1e-3
		@test f_proj.obs == obs_ans[proj_obs_indices, :]
		@test f_proj.var == data.var[data.var.name.>="D", :]

		f = filter_var(row->row.name>"D", data)
		@test materialize(f) ≈ X[data.var.name.>="D", :]
		@test f.obs == obs_ans
		@test f.var == data.var[data.var.name.>="D", :]
		f_proj = project(data_proj, f)
		@test materialize(f_proj) ≈ X[data.var.name.>="D", proj_obs_indices] rtol=1e-3
		@test f_proj.obs == obs_ans[proj_obs_indices, :]
		@test f_proj.var == data.var[data.var.name.>="D", :]

		f = filter_matrix("name"=>>("D"), 1:10:N, data)
		@test materialize(f) ≈ X[data.var.name.>="D", 1:10:N]
		@test f.obs == obs_ans[1:10:N, :]
		@test f.var == data.var[data.var.name.>="D", :]
		@test_throws ArgumentError project(data_proj, f)

		f = filter_matrix("name"=>>("D"), "group"=>==("A"), data)
		@test materialize(f) ≈ X[data.var.name.>="D", g.=="A"]
		@test f.obs == obs_ans[g.=="A", :]
		@test f.var == data.var[data.var.name.>="D", :]
		f_proj = project(data_proj, f)
		obs_ind = intersect(findall(g.=="A"),proj_obs_indices)
		@test materialize(f_proj) ≈ X[data.var.name.>="D", obs_ind] rtol=1e-3
		@test f_proj.obs == obs_ans[obs_ind, :]
		@test f_proj.var == data.var[data.var.name.>="D", :]
	end


	@testset "force layout seed=$seed" for seed in 1:5
		fl = force_layout(reduced; ndim=3, k=10, seed)
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

	@testset "pseudobulk $name" for (name,data,data_proj) in (("counts",counts,counts_proj), ("transformed",transformed,transformed_proj))
	# TODO: Make pseudobulk work with data.matrix::Factorization
	# @testset "pseudobulk $name" for (name,data,data_proj) in (("counts",counts,counts_proj), ("transformed",transformed,transformed_proj), ("reduced",reduced,reduced_proj))
		d = copy(data)
		d.obs.group2 = replace(d.obs.group, "C"=>missing)
		d.obs.group3 = rand(StableRNG(276), ("a","b"), size(d,2))
		d.obs.twogroup = replace(d.obs.group, "C"=>"A")
		X = materialize(d.matrix)

		d_proj = copy(data_proj)
		d_proj.obs.group2 = replace(d_proj.obs.group, "C"=>missing)
		d_proj.obs.group3 = rand(StableRNG(276), ("a","b"), size(d_proj,2))
		d_proj.obs.twogroup = replace(d_proj.obs.group, "C"=>"A")
		X_proj = materialize(d_proj.matrix)

		@testset "$annot" for annot in ("group","group2","group3","twogroup")
			unique_groups = collect(skipmissing(unique!(sort(d.obs[!,annot]))))

			pb = pseudobulk(d, annot)
			@test names(pb.obs) == ["id", annot]
			@test unique!(sort(pb.obs.id)) == unique_groups
			@test unique!(sort(pb.obs[!,annot])) == unique_groups

			pb_X = materialize(pb.matrix)
			@test size(pb_X,1) == size(X,1)
			@test size(pb_X,2) == length(unique_groups)

			for g in unique_groups
				x = vec(mean(X[:, isequal.(d.obs[!,annot], g)]; dims=2))
				gi = findfirst(isequal(g), pb.obs.id)
				@test x ≈ pb_X[:,gi]
			end


			unique_groups_proj = collect(skipmissing(unique!(sort(d_proj.obs[!,annot]))))

			pb_proj = project(d_proj, pb)
			@test names(pb_proj.obs) == ["id", annot]
			@test unique!(sort(pb_proj.obs.id)) == unique_groups_proj
			@test unique!(sort(pb_proj.obs[!,annot])) == unique_groups_proj

			pb_proj_X = materialize(pb_proj.matrix)
			@test size(pb_proj_X,1) == size(X_proj,1)
			@test size(pb_proj_X,2) == length(unique_groups_proj)

			for g in unique_groups_proj
				x_proj = vec(mean(X_proj[:, isequal.(d_proj.obs[!,annot], g)]; dims=2))
				gi = findfirst(isequal(g), pb_proj.obs.id)
				@test x_proj ≈ pb_proj_X[:,gi]
			end
		end

		@testset "$annot1, $annot2" for (annot1,annot2) in (("group","group3"),("group","group2"),("group","twogroup"))
			groups = string.(d.obs[!,annot1],'_',d.obs[!,annot2])
			mask = .!ismissing.(d.obs[!,annot1]) .& .!ismissing.(d.obs[!,annot2])
			unique_groups = unique!(sort!(groups[mask]))

			pb = pseudobulk(d, annot1, annot2)
			@test names(pb.obs) == ["id", annot1, annot2]
			@test unique!(sort(pb.obs.id)) == unique_groups
			@test unique!(sort(pb.obs[!,annot1])) == unique!(sort!(d.obs[mask,annot1]))
			@test unique!(sort(pb.obs[!,annot2])) == unique!(sort!(d.obs[mask,annot2]))

			pb_X = materialize(pb.matrix)
			@test size(pb_X,1) == size(X,1)
			@test size(pb_X,2) == length(unique_groups)

			for g1 in unique(d.obs[mask,annot1]), g2 in unique(d.obs[mask,annot2])
				g_mask = isequal.(d.obs[!,annot1], g1) .& isequal.(d.obs[!,annot2], g2)
				x = vec(mean(X[:, g_mask]; dims=2))
				gi = findfirst(isequal(string(g1,'_',g2)), pb.obs.id)
				if any(g_mask) # are there any observations in this group?
					@test x ≈ pb_X[:,gi]
				else
					@test gi === nothing
				end
			end


			groups_proj = string.(d_proj.obs[!,annot1],'_',d_proj.obs[!,annot2])
			mask_proj = .!ismissing.(d_proj.obs[!,annot1]) .& .!ismissing.(d_proj.obs[!,annot2])
			unique_groups_proj = unique!(sort!(groups_proj[mask_proj]))

			pb_proj = project(d_proj, pb)
			@test names(pb_proj.obs) == ["id", annot1, annot2]
			@test unique!(sort(pb_proj.obs.id)) == unique_groups_proj
			@test unique!(sort(pb_proj.obs[!,annot1])) == unique!(sort!(d_proj.obs[mask_proj,annot1]))
			@test unique!(sort(pb_proj.obs[!,annot2])) == unique!(sort!(d_proj.obs[mask_proj,annot2]))

			pb_proj_X = materialize(pb_proj.matrix)
			@test size(pb_proj_X,1) == size(X_proj,1)
			@test size(pb_proj_X,2) == length(unique_groups_proj)

			for g1 in unique(d_proj.obs[mask_proj,annot1]), g2 in unique(d_proj.obs[mask_proj,annot2])
				g_mask = isequal.(d_proj.obs[!,annot1], g1) .& isequal.(d_proj.obs[!,annot2], g2)
				x_proj = vec(mean(X_proj[:, g_mask]; dims=2))
				gi = findfirst(isequal(string(g1,'_',g2)), pb_proj.obs.id)
				if any(g_mask) # are there any observations in this group?
					@test x_proj ≈ pb_proj_X[:,gi]
				else
					@test gi === nothing
				end
			end
		end
	end
end
