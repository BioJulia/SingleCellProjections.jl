using Test
import SingleCellProjections as SCP
import Muon # loads MuonExt, so that `SCP.load_h5ad` below has methods

# A keyword that no public function accepts.
const BOGUS_KWARG = (; bogus_kwarg_xyz = 1)

# `umap`, `tsne` and `load_csv` forward their keywords to UMAP.jl, TSne.jl and CSV.jl, which have
# explicit keyword lists of their own, so an unknown keyword is rejected there - when the job is
# computed rather than when it is created. `load_h5ad` is swept: its keywords are consumed by our
# own `load_h5ad_matrix_impl`, Muon never sees them.
const KWARGS_EXEMPT = Set((:umap, :tsne, :load_csv))


"""
Jobs for the sweep to build its calls from. Only ever constructed, never fetched.
"""
function kwargs_test_fixtures()
	counts = SCP.load_counts(h5_path; sample_names="a")
	counts = SCP.add_obs_column(counts, "group", counts_obs_group)
	counts = SCP.add_obs_column(counts, "value", counts_obs_value)
	var     = SCP.get_var(counts)
	logt    = SCP.logtransform(counts)
	norm    = SCP.normalize_matrix(logt)
	reduced = SCP.pca(norm; nsv=4, niter=1)
	(; counts, var, logt, norm, reduced,
	   obs  = SCP.get_obs(counts),
	   fl   = SCP.force_layout(reduced; ndim=2, k=10, niter=1),
	   var2 = SCP.get_columns(var, "id", "name"), # transform_annotation wants exactly 2 columns
	)
end


# Every public function that creates a job. Each entry takes the extra keywords to pass, so the
# same call can be made both with and without the bogus one.
kwarg_sweep_calls(fx, h5ad_path) = [
	# --- load / transform ---
	:load_counts                  => kw -> SCP.load_counts(h5_path; sample_names="a", kw...),
	:logtransform                 => kw -> SCP.logtransform(fx.counts; kw...),
	:sctransform                  => kw -> SCP.sctransform(fx.counts; kw...),
	:tf_idf_transform             => kw -> SCP.tf_idf_transform(fx.counts; kw...),

	# --- normalize ---
	:normalize_matrix             => kw -> SCP.normalize_matrix(fx.logt; kw...),
	:negative_regression_matrix   => kw -> SCP.negative_regression_matrix(fx.logt, SCP.designmatrix(fx.logt); kw...),
	:designmatrix                 => kw -> SCP.designmatrix(fx.logt; kw...),

	# --- reduce ---
	:svd                          => kw -> SCP.svd(fx.norm; nsv=4, niter=1, kw...),
	:pca                          => kw -> SCP.pca(fx.norm; nsv=4, niter=1, kw...),
	:loadings                     => kw -> SCP.loadings(fx.norm; nsv=4, niter=1, kw...),
	:force_layout                 => kw -> SCP.force_layout(fx.reduced; ndim=2, k=10, niter=1, kw...),

	# --- annotate ---
	:annotate_var                 => kw -> SCP.annotate_var(fx.counts, fx.var; kw...),
	:annotate_obs                 => kw -> SCP.annotate_obs(fx.counts, fx.obs; kw...),
	:var_counts_sum               => kw -> SCP.var_counts_sum(fx.counts, "vcs"; kw...),
	:obs_counts_sum               => kw -> SCP.obs_counts_sum(fx.counts, "ocs"; kw...),
	:var_counts_fraction          => kw -> SCP.var_counts_fraction(fx.counts, "vcf", "name"=>startswith("A"); kw...),
	:obs_counts_fraction          => kw -> SCP.obs_counts_fraction(fx.counts, "ocf", "barcode"=>!isequal(""); kw...),

	# --- filter ---
	:filter_var                   => kw -> SCP.filter_var("name"=>startswith("A"), fx.counts; kw...),
	:filter_obs                   => kw -> SCP.filter_obs("barcode"=>!isequal(""), fx.counts; kw...),
	:filter_matrix                => kw -> SCP.filter_matrix("name"=>startswith("A"), "barcode"=>!isequal(""), fx.counts; kw...),

	# --- sum of squares ---
	:variance                     => kw -> SCP.variance(fx.norm; assume_centered=true, kw...),
	:std                          => kw -> SCP.std(fx.norm; assume_centered=true, kw...),
	:relative_std                 => kw -> SCP.relative_std(fx.norm; assume_centered=true, kw...),

	# --- tables ---
	:get_colnames                 => kw -> SCP.get_colnames(fx.var; kw...),
	:get_columns                  => kw -> SCP.get_columns(fx.var, "id"; kw...),
	:column_data                  => kw -> SCP.column_data(fx.var, "id"; kw...),
	:transform_annotation         => kw -> SCP.transform_annotation(identity, fx.var2; kw...),

	# --- statistical tests ---
	:ftest                        => kw -> SCP.ftest(fx.logt, "group"; kw...),
	:ttest                        => kw -> SCP.ttest(fx.logt, "value"; kw...),
	:mannwhitney                  => kw -> SCP.mannwhitney(fx.logt, "group", "A", "B"; kw...),

	# --- misc ---
	:project                      => kw -> SCP.project(fx.norm, fx.counts; kw...),
	:signature                    => kw -> SCP.signature(fx.norm, "name"=>startswith("A"), "sig"; kw...),
	:transform_coords             => kw -> SCP.transform_coords(fx.fl, SCP.rot2d(0.1); kw...),
	:find_optimal_coord_transform => kw -> SCP.find_optimal_coord_transform(fx.fl, "group"=>isequal("A"), "group"=>isequal("B"); kw...),
	:pseudobulk                   => kw -> SCP.pseudobulk(fx.logt, "group"; kw...),
	:population_matrix            => kw -> SCP.population_matrix(fx.obs, "sample_name"; new_var_covariates="group", kw...),
	:transfer_annotation          => kw -> SCP.transfer_annotation(fx.reduced, fx.reduced, "group"; k=5, kw...),

	# --- extensions ---
	:load_h5ad                    => kw -> SCP.load_h5ad(h5ad_path; kw...),
]


# `Base.kwarg_decl` reports a `kwargs...` slurp as the symbol `kwargs...`; a bare `...` instead
# means an auto-generated method for a default positional argument, which is not what we want.
_has_kwargs_slurp(f) =
	any(m->any(k->(s = String(k); endswith(s,"...") && s != "..."), Base.kwarg_decl(m)), methods(f))

"""
Public functions with a `kwargs...` slurp, i.e. the ones that can silently swallow a keyword
unless they check. Derived from the module rather than listed, so a newly added one shows up.
"""
public_kwargs_functions() =
	Set(n for n in names(SCP) if n != nameof(SCP) && isdefined(SCP, n) &&
	    getfield(SCP, n) isa Function && _has_kwargs_slurp(getfield(SCP, n)))


function run_kwargs_tests()
	@testset "Public kwargs" begin
		# An empty file is enough: the sweep only ever constructs jobs, never fetches them, so
		# nothing reads the contents - `checksummedfilepath_job` just needs the path to exist.
		h5ad_path = joinpath(mktempdir(), "empty.h5ad")
		touch(h5ad_path)
		calls = kwarg_sweep_calls(kwargs_test_fixtures(), h5ad_path)

		# An unknown keyword must be rejected when the job is *created*, not when it is computed.
		# Otherwise it silently becomes part of the job spec: the hash changes, the computation
		# does not, and a misspelled keyword just invalidates the cache.
		@testset "$name" for (name, call) in calls
			# The control call is what makes the check below mean anything: it pins down that the
			# arguments are fine on their own, so the failure is caused by the keyword. Without
			# it the test could pass on an argument error - a MethodError message quotes the
			# whole call, bogus keyword included.
			@test call(NamedTuple()) !== nothing
			@test_throws "bogus_kwarg_xyz" call(BOGUS_KWARG)
		end

		# The check is opt-in, so a new public function can forget it. Fail here if one appears.
		@testset "coverage" begin
			unswept = setdiff(public_kwargs_functions(), Set(first.(calls)), KWARGS_EXEMPT)
			@test sort!(collect(unswept)) == Symbol[]
		end
	end
end
