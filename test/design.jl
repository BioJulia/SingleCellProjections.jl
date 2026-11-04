using Test
using SingleCellProjections
import SingleCellProjections as SCP
using ReproducibleJobs: fetch!
using LinearAlgebra


function run_design_tests()
	@testset "Design matrix" begin
		n = 20
		counts = SCP.load_counts(h5_path; sample_names="a")
		counts = SCP.filter_obs(1:n, counts)
		counts = SCP.add_obs_column(counts, "cell", string.("cell_", 1:n))

		@testset "one category per observation" begin
			design = fetch!(SCP.get_matrix(SCP.designmatrix(counts, "cell")))
			@test design == hcat(ones(n), I(n)) # intercept and one column per category
		end
	end
end
