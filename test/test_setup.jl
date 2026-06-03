__revise_mode__ = :evalassign

# --- test_utils ---

isgz(fn) = lowercase(splitext(fn)[2])==".gz"
_open(f, fn) = open(fn) do io
    f(isgz(fn) ? GzipDecompressorStream(io) : io)
end

read_matrix(fn,delim=',') = _open(io->readdlm(io,delim,Int), fn)
read_strings(fn,delim=',') = _open(io->readdlm(io,delim,String), fn)


# NB: DataFrame columns are compared with ===
function test_dataframe_columns_identical(context, a, b)
	isequal(names(a), names(b)) || @warn "DataFrame comparison $context failed."
	@test isequal(names(a), names(b))

	for n in names(a)
		a[!,n] === b[!,n] || @warn "DataFrame comparison $context failed for column \"$n\":"
		@test a[!,n] === b[!,n]
	end
end


function simple_logtransform(X, scale_factor)
	s = sum(X; dims=1)
	log2.( 1 .+ X.*scale_factor./max.(1,s) )
end

function simple_idf(X)
	df = vec(max.(mean(X; dims=2), 1/size(X,2)))
	1 ./ df
end

function simple_tf_idf_transform(X, idf, scale_factor)
	@assert size(idf) == (size(X,1),)
	X = Matrix(X)
	# log( 1 + c*f/s * idf )
	s = max.(1,sum(X;dims=1))
	sparse(log.(1 .+ X.*scale_factor.*idf./s))
end


materialize(X) = X
materialize(X::MatrixExpression) = X*I(size(X,2))
materialize(X::SVD) = convert(Matrix,X)
# materialize(X::PMA) = X.U*Diagonal(X.S)*X.Vt
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

function test_show(data::DataMatrix; matrix=nothing, var=nothing, obs=nothing)
    io = IOBuffer()
    show(io, MIME("text/plain"), data)
    str = String(take!(io))
    s = split(str, '\n')
    @test 4<=length(s)<=5
    @test s[1] == "DataMatrix ($(size(data,1)) variables and $(size(data,2)) observations)"
    matrix!==nothing && @test contains(s[2][3:end], matrix)
    @test startswith(s[3],"  Variables: ")
    var!==nothing && @test sort(split(s[3][14:end], ", "))==sort(var)
    @test startswith(s[4],"  Observations: ")
    obs!==nothing && @test sort(split(s[4][17:end], ", "))==sort(obs)
end

function _formula(args...; center=true)
	if center
		StatsModels.Term(:y) ~ +(StatsModels.ConstantTerm(1), StatsModels.Term.(Symbol.(args))...)
	elseif !isempty(args)
		StatsModels.Term(:y) ~ sum(StatsModels.Term.(Symbol.(args)))
	else
		nothing # Cannot create formula without RHS (i.e. only zero)
	end
end


function ttest_ground_truth(A, obs, formula, group_a; center)
	t = zeros(size(A,1))
	p = zeros(size(A,1))
	β = zeros(size(A,1))

	df = copy(obs)
	df.y = zeros(size(A,2))
	for i in 1:size(A,1)
		df.y = A[i,:]

		if center
			m = lm(formula, df)
		else
			m = lm(modelmatrix(formula,df), df.y) # this skips the intercept
		end

		table = coeftable(m)
		@assert table.colnms[1] == "Coef."

		t[i] = table.cols[table.teststatcol][end]
		p[i] = table.cols[table.pvalcol][end]
		β[i] = table.cols[1][end]

		# a little hack since we cannot control which group is outputted by lm/coeftable
		if group_a !== nothing && !endswith(table.rownms[end], ": $group_a")
			t[i] = -t[i]
			β[i] = -β[i]
		end
	end

	t,p,β
end
function ttest_ground_truth(A, obs, h1, group_a, group_b, h0::Tuple; center=true)
	h1 in h0 && return zeros(size(A,1)), ones(size(A,1)), zeros(size(A,1))

	if !(eltype(obs[!,h1]) <: Union{Missing,Number})
		if group_a === nothing
			group_a = first(sort(obs[!,h1]))
		elseif group_b === nothing # overwrite everything except a
			obs = copy(obs)
			obs[.!isequal.(obs[!,h1],group_a), h1] .= "Not_$group_a"
		else # keep only observations belonging to group a and group b
			mask = isequal.(obs[!,h1],group_a) .| isequal.(obs[!,h1],group_b)
			A = A[:,mask]
			obs = obs[mask, :]
		end
	end

	formula = _formula(h0..., h1; center)
	return ttest_ground_truth(A, obs, formula, group_a; center)
end
ttest_ground_truth(A, obs, h1, group_a, h0::Tuple; kwargs...) = ttest_ground_truth(A,obs,h1,group_a,nothing,h0; kwargs...)
ttest_ground_truth(A, obs, h1, h0::Tuple; kwargs...) = ttest_ground_truth(A,obs,h1,nothing,nothing,h0; kwargs...)


function ftest_ground_truth(A, obs, h1_formula, h0_formula)
	F = zeros(size(A,1))
	p = zeros(size(A,1))

	df = copy(obs)
	df.y = zeros(size(A,2))
	for i in 1:size(A,1)
		df.y = A[i,:]
		h0 = lm(h0_formula, df).model # This includes intercept by default.
		h1 = lm(h1_formula, df).model # (Not posible to turn off?)
		ft = GLM.ftest(h0, h1)
		F[i] = ft.fstat[end]
		p[i] = ft.pval[end]
	end

	F,p
end
function ftest_ground_truth(A, obs, h1::Tuple, h0::Tuple)
	# simple unwrapping of Covariates, does not care about types or two-groups
	h1 = (x->x isa CovariateDesc ? x.src : x).(h1)
	h0 = (x->x isa CovariateDesc ? x.src : x).(h0)

	all(in(h0), h1) && return zeros(size(A,1)), ones(size(A,1))

	h1_formula = _formula(h0..., h1...)
	h0_formula = _formula(h0...)
	return ftest_ground_truth(A, obs, h1_formula, h0_formula)
end


# --- common_data ---

# Paths
pbmc_path = joinpath(pkgdir(SingleCellProjections), "test/data/500_PBMC_3p_LT_Chromium_X_50genes")
h5_path = joinpath(pbmc_path, "filtered_feature_bc_matrix.h5")
mtx_path = joinpath(pbmc_path, "filtered_feature_bc_matrix/matrix.mtx.gz")

h5_subset_path = joinpath(pbmc_path, "filtered_feature_bc_matrix_272barcodes.h5")
pbmc_subset_barcodes = read_strings(joinpath(pbmc_path,"barcodes_272.csv"))[2:end]


rna_adt_h5_path = joinpath(pkgdir(SingleCellProjections), "test/data/GSE164378_RNA_ADT_3P_P1_subsetted.h5")
rna_h5_path = joinpath(pkgdir(SingleCellProjections), "test/data/GSE164378_RNA_3P_P1_subsetted.h5")

# Ground truth
expected_mat = read_matrix(joinpath(pbmc_path,"expected_matrix.csv"))
expected_nnz = count(!iszero, expected_mat)
expected_feature_ids = vec(read_strings(joinpath(pbmc_path,"expected_feature_ids.csv")))
expected_barcodes = vec(read_strings(joinpath(pbmc_path,"expected_barcodes.csv")))

expected_feature_names = read_strings(joinpath(pbmc_path,"filtered_feature_bc_matrix/features.tsv.gz"),'\t')[:,2]
expected_feature_types = fill("Gene Expression", 50)
expected_feature_genome = fill("GRCh38", 50)

expected_var = DataFrame("id"=>expected_feature_ids, "name"=>expected_feature_names, "feature_type"=>expected_feature_types, "genome"=>expected_feature_genome)

expected_sparse = sparse(expected_mat)


pbmc_subset_ind = indexin(pbmc_subset_barcodes, expected_barcodes)



params = SCTransform.scparams(expected_sparse, DataFrame(id=expected_feature_ids, name=expected_feature_names, feature_type=expected_feature_types); use_cache=false)

# # Data shared between tests
# counts = load10x(h5_path)

# counts.obs.group = rand(StableRNG(904), ("A","B","C"), size(counts,2))
# counts.obs.value = 1 .+ randn(StableRNG(905), size(counts,2))
counts_obs_group = rand(StableRNG(904), ("A","B","C"), length(expected_barcodes))
counts_obs_value = 1 .+ randn(StableRNG(905), length(expected_barcodes))

# transformed = sctransform(counts; use_cache=false)
# normalized = normalize_matrix(transformed, "group", "value")
# reduced = svd(normalized; nsv=10, niter=4, rng=StableRNG(102))
