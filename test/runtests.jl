# To run an individual test suite from the REPL:
# 1. Activate the test environment
# 2. includet("test/test_setup.jl")
#    includet("test/load.jl")
# 3. with_scheduler(Scheduler(; dir=mktempdir())) do
#        register_scp_functions!()
#        run_load_tests()
#    end

include("test_setup.jl")
include("projectables.jl")
include("tables.jl")
include("load.jl")
include("transform.jl")
include("reduce.jl")
include("filter.jl")
include("subset.jl")
include("sum_squared.jl")
include("umap.jl")
include("tsne.jl")
include("muon.jl")
include("SCPCore/runtests.jl")
include("MatrixExpressions/runtests.jl")

# mktempdir() do tmp # Cleanup directly
let tmp = mktempdir() # Cleanup when Julia process exits - useful for inspecting
	@testset "SingleCellProjections.jl" begin
		# The 2nd time we run, the on-disk cache is reused. (Consider doing this kind of cache testing in ReproducibleJobs.jl only.)
		@testset "$cache_status" for cache_status in ("New Disk Cache", "Reused Disk Cache")
		# @testset "$cache_status" for cache_status in ("New Disk Cache",)
			with_scheduler(Scheduler(; dir=tmp)) do
				register_scp_functions!()
				run_projectables_tests()
				run_tables_tests()
				run_load_tests()
				run_transform_tests()
				run_reduce_tests()
				run_filter_tests()
				run_subset_tests()
				run_sum_squared_tests()
				run_umap_tests()
				run_tsne_tests()
				run_muon_tests()
			end
		end
		run_core_tests()
		run_matrix_expressions_tests()
	end
end
