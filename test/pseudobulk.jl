using Test
using SingleCellProjections
import SingleCellProjections as SCP
using SingleCellProjections: SCPCore
using .SCPCore: unblockify
using ReproducibleJobs: fetch!
using DataFrames


# The mean of the cells matching each row of `pb_obs` in the covariates `covs`.
function pseudobulk_ground_truth(X, obs, pb_obs, covs)
	reduce(hcat, map(eachrow(pb_obs)) do row
		mask = trues(size(X,2))
		for c in covs
			mask .&= obs[!,c] .== row[c]
		end
		any(mask) ? vec(sum(X[:,mask]; dims=2)) ./ count(mask) : zeros(size(X,1))
	end)
end

# The fraction of the cells in each obs category (`obs_cov`) that belong to each var category (`var_cov`).
function population_matrix_ground_truth(obs, pm_var, pm_obs, var_cov, obs_cov)
	[count((obs[!,var_cov] .== v) .& (obs[!,obs_cov] .== o)) / count(obs[!,obs_cov] .== o)
	 for v in pm_var[!,var_cov], o in pm_obs[!,obs_cov]]
end


function run_pseudobulk_tests()
	@testset "Pseudobulk" begin
		n = 20
		counts = SCP.load_counts(h5_path; sample_names="a")
		counts = SCP.filter_obs(1:n, counts)
		counts = SCP.add_obs_column(counts, "group", counts_obs_group[1:n])
		counts = SCP.add_obs_column(counts, "cell", string.("cell_", 1:n)) # one category per observation

		c = fetch!(counts)
		X = convert(Matrix{Float64}, unblockify(materialize(c)))
		obs = c.obs

		@testset "pseudobulk $covs" for covs in (("group",), ("cell",), ("cell","group"), ("group","cell"))
			pb = fetch!(SCP.pseudobulk(counts, covs...))
			pb_X = convert(Matrix{Float64}, unblockify(materialize(pb)))
			@test pb_X ≈ pseudobulk_ground_truth(X, obs, pb.obs, covs)
		end

		@testset "population_matrix obs:$obs_cov var:$var_cov" for (obs_cov,var_cov) in (("group","cell"), ("cell","group"))
			pm = fetch!(SCP.population_matrix(SCP.get_obs(counts), obs_cov; new_var_covariates=var_cov))
			@test pm.matrix ≈ population_matrix_ground_truth(obs, pm.var, pm.obs, var_cov, obs_cov)
		end
	end
end
