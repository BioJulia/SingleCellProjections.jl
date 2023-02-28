@testset "DataMatrix" begin
	d = DataMatrix()
	@test size(d) == (0,0)
	@test d.var == DataFrame(;id=String[]) == d.obs
	@test d.var_id_cols == ["id"] == d.obs_id_cols
	@test string(d) == "0×0 Matrix{Float64}"

	d = DataMatrix([11 21; 12 22; 13 23], DataFrame(id=["A","B","C"]), DataFrame(id=["u","v"]))
	@test size(d) == (3,2)
	@test d.matrix == [11 21; 12 22; 13 23]
	@test d.var == DataFrame(;id=String["A","B","C"])
	@test d.obs == DataFrame(;id=String["u","v"])
	@test d.var_id_cols == ["id"] == d.obs_id_cols
	@test string(d) == "3×2 Matrix{$Int}"
	test_show(d; matrix="Matrix{$Int}", var=["id"], obs=["id"], models=r"^$")

	push!(d.obs_id_cols, "bad")
	test_show(d; matrix="Matrix{$Int}", var=["id"], obs=["id","bad"], models=r"^$")

	@test_throws DimensionMismatch DataMatrix([11 21; 12 22; 13 23], DataFrame(id=["A","B"]), DataFrame(id=["u","v"]))
	@test_throws DimensionMismatch DataMatrix([11 21; 12 22; 13 23], DataFrame(id=["A","B","C"]), DataFrame(id=["u","v","w"]))
end
