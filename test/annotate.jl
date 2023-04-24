@testset "Annotate" begin
	@testset "var_to_obs" begin
		c = copy(counts)
		n = size(c.obs,2)

		table = var_to_obs_table(:name=>==("GPR22"), c; names="OutName")
		@show names(table)
		@test table.OutName == c.matrix[findfirst(==("GPR22"),c.var.name),:]
		@test eltype(table.OutName) <: Integer

		var_to_obs!(:name=>==("GPR22"), c; names="OutName")
		n += 1
		@test c.obs.OutName == c.matrix[findfirst(==("GPR22"),c.var.name),:]
		@test eltype(c.obs.OutName) <: Integer
		@test size(c.obs,2) == n

		@test_throws ArgumentError var_to_obs!(:name=>==("GPR22"), c; names="OutName")
		@test size(c.obs,2) == n

		var_to_obs!(:name=>==("MALAT1"), c)
		n += 1
		@test c.obs.MALAT1 == c.matrix[findfirst(==("MALAT1"),c.var.name),:]
		@test size(c.obs,2) == n

		var_to_obs!(:name=>==("MALAT1"), c; name_src=:id)
		n += 1
		@test c.obs.ENSG00000251562 == c.matrix[findfirst(==("MALAT1"),c.var.name),:]
		@test size(c.obs,2) == n

		var_to_obs!(:name=>startswith("F"), c)
		n += 2
		@test c.obs.FBN2 == c.matrix[findfirst(==("FBN2"),c.var.name),:]
		@test c.obs.FTL == c.matrix[findfirst(==("FTL"),c.var.name),:]
		@test size(c.obs,2) == n

		var_to_obs!(:name=>startswith("F"), c; names=["OutA","OutB"])
		n += 2
		i1,i2 = minmax(findfirst(==("FBN2"),c.var.name), findfirst(==("FTL"),c.var.name))
		@test c.obs.OutA == c.matrix[i1,:]
		@test c.obs.OutB == c.matrix[i2,:]
		@test size(c.obs,2) == n

		var_to_obs!([3], c)
		n += 1
		@test c.obs[:,"ENSG00000184924_Gene Expression"] == c.matrix[3,:]
		@test size(c.obs,2) == n

		@test_throws ArgumentError var_to_obs!([3], c; name_src=:id, names="OutC")
		@test size(c.obs,2) == n

		t2 = var_to_obs(:name=>==("GPR22"), normalized)
		@test t2.obs.GPR22 ≈ materialize(normalized)[findfirst(==("GPR22"),normalized.var.name), :]
		@test eltype(t2.obs.GPR22) <: Float64
	end

	proj_var_ind = reverse(1:2:size(counts,1))
	proj_obs_ind = 1:12:size(counts,2)
	counts_proj = counts[proj_var_ind,proj_obs_ind]
	empty!(counts_proj.models)

	@testset "var_to_obs projection" begin
		c = var_to_obs(:name=>==("GPR22"), counts)
		c_proj = project(counts_proj,c)
		@test c_proj.obs.GPR22 == counts_proj.matrix[findfirst(==("GPR22"),counts_proj.var.name),:]
		@test eltype(c_proj.obs.GPR22) <: Integer
		@test size(c_proj.obs,2) == size(counts_proj.obs,2)+1

		c = var_to_obs(:name=>startswith("C"), counts)
		c_proj = project(counts_proj,c)
		@test c_proj.obs.CFAP298 == counts_proj.matrix[findfirst(==("CFAP298"),counts_proj.var.name),:]
		@test all(ismissing, c_proj.obs.C3orf79)
		@test all(ismissing, c_proj.obs.CDY1)
		@test all(ismissing, c_proj.obs.CEBPA)
		@test all(ismissing, c_proj.obs.CST3)

		@test eltype(c_proj.obs.CFAP298) <: Integer
		@test typeof(c_proj.obs.C3orf79) <: Vector{Union{Missing,T}} where T<:Integer
		@test typeof(c_proj.obs.CDY1)    <: Vector{Union{Missing,T}} where T<:Integer
		@test typeof(c_proj.obs.CEBPA)   <: Vector{Union{Missing,T}} where T<:Integer
		@test typeof(c_proj.obs.CST3)    <: Vector{Union{Missing,T}} where T<:Integer

		@test size(c_proj.obs,2) == size(counts_proj.obs,2)+5
	end

end
