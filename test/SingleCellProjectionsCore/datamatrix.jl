@testset "DataMatrix" begin
	@testset "Basic" begin
		d = DataMatrix()
		@test size(d) == (0,0)
		@test d.var == DataFrame(;id=String[]) == d.obs
		@test names(d.var,1) == ["id"] == names(d.obs,1)
		@test string(d) == "0×0 Matrix{Float64}"

		mat = [11 21 31 41; 12 22 32 42; 13 23 33 43]
		var_annot = DataFrame(id=["A","B","C"], id2=["Z","Y","X"], v1=["A","A","B"], v2=["1","2","2"])
		obs_annot = DataFrame(id=["a","b","c","d"], o1=["A","A","B","B"], o2=["1","2","1","2"])

		d = DataMatrix(mat, var_annot, obs_annot)
		@test size(d) == (3,4)
		@test d.matrix == [11 21 31 41; 12 22 32 42; 13 23 33 43]
		@test d.var == DataFrame(id=["A","B","C"], id2=["Z","Y","X"], v1=["A","A","B"], v2=["1","2","2"])
		@test d.obs == DataFrame(id=["a","b","c","d"], o1=["A","A","B","B"], o2=["1","2","1","2"])
		@test names(d.var,1) == ["id"] == names(d.obs,1)
		@test string(d) == "3×4 Matrix{$Int}"
		# test_show(d; matrix="Matrix{$Int}", var=["id","id2","v1","v2"], obs=["id","o1","o2"])

		@test_throws ErrorException DataMatrix(mat, var_annot[:,[:v1]], obs_annot; duplicate_var=:error)
		d = DataMatrix(mat, var_annot[:,[:v1]], obs_annot; duplicate_var=:ignore)
		@test names(d.var,1) == ["v1"]
		d = DataMatrix(mat, var_annot[:,[:v2]], obs_annot)#; duplicate_var=:warn)
		@test names(d.var,1) == ["v2"]

		@test_throws ErrorException DataMatrix(mat, var_annot, obs_annot[:,[:o1]]; duplicate_obs=:error)
		d = DataMatrix(mat, var_annot, obs_annot[:,[:o1]]; duplicate_var=:ignore)
		@test names(d.obs,1) == ["o1"]
		d = DataMatrix(mat, var_annot, obs_annot[:,[:o2]])#; duplicate_var=:warn)
		@test names(d.obs,1) == ["o2"]

		@test_throws DimensionMismatch DataMatrix([11 21; 12 22; 13 23], DataFrame(id=["A","B"]), DataFrame(id=["u","v"]))
		@test_throws DimensionMismatch DataMatrix([11 21; 12 22; 13 23], DataFrame(id=["A","B","C"]), DataFrame(id=["u","v","w"]))
	end
end
