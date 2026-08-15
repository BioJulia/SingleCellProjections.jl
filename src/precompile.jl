using PrecompileTools

# A standard pipeline, run at build time so that users do not pay its compilation on every first
# call. Kept deliberately small: every call added here costs precompilation time for everyone, so
# extend it only in response to `--trace-compile-timing` evidence that something is still being
# compiled at runtime.
#
# NB: `__init__` does not run during precompilation, so the global scheduler is not set up here --
# the workload has to create its own (with a temporary cache dir) and register the functions itself.
@setup_workload begin
	h5_path = joinpath(pkgdir(@__MODULE__), "test", "data",
	                   "500_PBMC_3p_LT_Chromium_X_50genes", "filtered_feature_bc_matrix.h5")

	@compile_workload begin
		redirect_stdout(devnull) do # the scheduler's progress display would spam the build log
			mktempdir() do dir
				ReproducibleJobs.with_scheduler(ReproducibleJobs.Scheduler(; dir)) do
					register_scp_functions!()

					raw_counts = load_counts([h5_path, h5_path]; sample_names=["a", "b"])
					# This file has no "MT-" genes; use a filter that actually matches.
					counts = var_counts_fraction(raw_counts, "fraction_a", "name" => startswith("A"))
					counts = filter_obs("fraction_a" => <(0.03), counts)

					transformed = sctransform(counts)
					normalized = normalize_matrix(transformed)
					reduced = pca(normalized; nsv=4, niter=1)
					fl = force_layout(reduced; ndim=2, k=10, niter=1)

					fetch!(counts)
					fetch!(fl)

					fetch!(logtransform(counts))

					normalized2 = normalize_matrix(transformed, "fraction_a")
					fetch!(normalized2)

					raw_counts_proj = load_counts(h5_path; sample_names="p")
					fetch!(project(fl, raw_counts=>raw_counts_proj))
					fetch!(project(normalized2, raw_counts=>raw_counts_proj))
				end
			end
		end
	end
end
