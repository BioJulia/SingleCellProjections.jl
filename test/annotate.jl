@testset "Annotate" begin
	@testset "var_to_obs" begin
		c = DataMatrix(counts.matrix, copy(counts.var), copy(counts.obs)) # TODO: copy(counts)
		n = size(c.obs,2)

		var_to_obs!(:name=>==("GPR22"), c; names="OutName")
		n += 1
		@test c.obs.OutName == c.matrix[findfirst(==("GPR22"),c.var.name),:]
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
	end
end
