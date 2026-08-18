using Test
import SingleCellProjections as SCP

# A keyword that no public function accepts.
# It must NOT start with `__`: ReproducibleJobs strips `__`-prefixed keywords before calling the
# function (that is the `__version` mechanism), so a `__`-prefixed probe would test that instead.
const BOGUS_KWARG = (; bogus_kwarg_xyz = 1)

# Extension functions are included in the sweep, but only once their package is loaded. Before
# that they exist as declarations with no methods, so there is no keyword check to exercise.
const KWARGS_EXEMPT = Set{Symbol}()


# Every public function that creates a job, called with the bogus keyword.
#
# The arguments are dummies. The keyword check is the first thing in the function body, and the
# arguments are unevaluated job specs, so the call must fail on the keyword long before anything
# looks at them. Required keywords *are* supplied, since Julia checks those at the call site.
kwarg_sweep_calls() = [
	# --- load / transform ---
	:load_counts                  => () -> SCP.load_counts(nothing; sample_names="a", BOGUS_KWARG...),
	:logtransform                 => () -> SCP.logtransform(nothing; BOGUS_KWARG...),
	:sctransform                  => () -> SCP.sctransform(nothing; BOGUS_KWARG...),
	:tf_idf_transform             => () -> SCP.tf_idf_transform(nothing; BOGUS_KWARG...),

	# --- normalize ---
	:normalize_matrix             => () -> SCP.normalize_matrix(nothing; BOGUS_KWARG...),
	:negative_regression_matrix   => () -> SCP.negative_regression_matrix(nothing, nothing; BOGUS_KWARG...),
	:designmatrix                 => () -> SCP.designmatrix(nothing; BOGUS_KWARG...),

	# --- reduce ---
	:svd                          => () -> SCP.svd(nothing; nsv=2, BOGUS_KWARG...),
	:pca                          => () -> SCP.pca(nothing; nsv=2, BOGUS_KWARG...),
	:loadings                     => () -> SCP.loadings(nothing; nsv=2, BOGUS_KWARG...),
	:force_layout                 => () -> SCP.force_layout(nothing; BOGUS_KWARG...),

	# --- annotate ---
	:annotate_var                 => () -> SCP.annotate_var(nothing, nothing; BOGUS_KWARG...),
	:annotate_obs                 => () -> SCP.annotate_obs(nothing, nothing; BOGUS_KWARG...),
	:var_counts_sum               => () -> SCP.var_counts_sum(nothing, "col"; BOGUS_KWARG...),
	:obs_counts_sum               => () -> SCP.obs_counts_sum(nothing, "col"; BOGUS_KWARG...),
	:var_counts_fraction          => () -> SCP.var_counts_fraction(nothing, "col", nothing; BOGUS_KWARG...),
	:obs_counts_fraction          => () -> SCP.obs_counts_fraction(nothing, "col", nothing; BOGUS_KWARG...),

	# --- filter ---
	:filter_var                   => () -> SCP.filter_var(nothing, nothing; BOGUS_KWARG...),
	:filter_obs                   => () -> SCP.filter_obs(nothing, nothing; BOGUS_KWARG...),
	:filter_matrix                => () -> SCP.filter_matrix(nothing, nothing, nothing; BOGUS_KWARG...),

	# --- sum of squares ---
	:variance                     => () -> SCP.variance(nothing; assume_centered=true, BOGUS_KWARG...),
	:std                          => () -> SCP.std(nothing; assume_centered=true, BOGUS_KWARG...),
	:relative_std                 => () -> SCP.relative_std(nothing; assume_centered=true, BOGUS_KWARG...),

	# --- tables ---
	:get_colnames                 => () -> SCP.get_colnames(nothing; BOGUS_KWARG...),
	:get_columns                  => () -> SCP.get_columns(nothing, "col"; BOGUS_KWARG...),
	:column_data                  => () -> SCP.column_data(nothing, "col"; BOGUS_KWARG...),
	:transform_annotation         => () -> SCP.transform_annotation(identity, nothing; BOGUS_KWARG...),

	# --- statistical tests ---
	:ftest                        => () -> SCP.ftest(nothing, "h1"; BOGUS_KWARG...),
	:ttest                        => () -> SCP.ttest(nothing, "h1"; BOGUS_KWARG...),
	:mannwhitney                  => () -> SCP.mannwhitney(nothing, "col"; BOGUS_KWARG...),

	# --- misc ---
	:project                      => () -> SCP.project(nothing, nothing; BOGUS_KWARG...),
	:signature                    => () -> SCP.signature(nothing, nothing, "col"; BOGUS_KWARG...),
	:transform_coords             => () -> SCP.transform_coords(nothing, nothing; BOGUS_KWARG...),
	:find_optimal_coord_transform => () -> SCP.find_optimal_coord_transform(nothing, nothing; BOGUS_KWARG...),
	:pseudobulk                   => () -> SCP.pseudobulk(nothing, "cov"; BOGUS_KWARG...),
	:population_matrix            => () -> SCP.population_matrix(nothing, "cov"; new_var_covariates="x", BOGUS_KWARG...),
	:transfer_annotation          => () -> SCP.transfer_annotation(nothing, nothing, "cov"; k=5, BOGUS_KWARG...),

	# --- extensions (only meaningful once the corresponding package is loaded) ---
	:umap                         => () -> SCP.umap(nothing; ndim=2, BOGUS_KWARG...),
	:tsne                         => () -> SCP.tsne(nothing; BOGUS_KWARG...),
	:load_csv                     => () -> SCP.load_csv("no_such_file.csv"; BOGUS_KWARG...),
	:load_h5ad                    => () -> SCP.load_h5ad(nothing; BOGUS_KWARG...),
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
		calls = kwarg_sweep_calls()

		# An unknown keyword must be rejected when the job is *created*, not when it is computed.
		# Otherwise it silently becomes part of the job spec: the hash changes, the computation
		# does not, and a misspelled keyword just invalidates the cache.
		@testset "$name" for (name, call) in calls
			@test_throws "bogus_kwarg_xyz" call()
		end

		# The check is opt-in, so a new public function can forget it. Fail here if one appears.
		@testset "coverage" begin
			unswept = setdiff(public_kwargs_functions(), Set(first.(calls)), KWARGS_EXEMPT)
			@test sort!(collect(unswept)) == Symbol[]
		end
	end
end
