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
		test_show(d; matrix="Matrix{$Int}", var=["id","id2","v1","v2"], obs=["id","o1","o2"], models=r"^$")

		set_var_id_col!(d, "id2")
		@test names(d.var,1) == ["id2"]
		@test_throws ArgumentError set_var_id_col!(d, "nonexisting")
		@test_throws ErrorException set_var_id_col!(d, "v1")
		@test_throws ErrorException set_var_id_col!(d, "v2")

		set_var_id_col!(d, "v1"; duplicate_var=:ignore)
		@test names(d.var,1) == ["v1"]
		set_var_id_col!(d, "v2"; duplicate_var=:warn)
		@test names(d.var,1) == ["v2"]

		set_var_id_col!(d, "id")
		@test names(d.var,1) == ["id"]

		@test_throws ArgumentError set_obs_id_col!(d, "nonexisting")
		@test_throws ErrorException set_obs_id_col!(d, "o1")
		@test_throws ErrorException set_obs_id_col!(d, "o2")

		set_obs_id_col!(d, "o1"; duplicate_obs=:ignore)
		@test names(d.obs,1) == ["o1"]
		set_obs_id_col!(d, "o2"; duplicate_obs=:warn)
		@test names(d.obs,1) == ["o2"]

		set_obs_id_col!(d, "id")
		@test names(d.obs,1) == ["id"]


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

	@testset "copy" begin
		d = DataMatrix([11 21 31 41; 12 22 32 42; 13 23 33 43],
		               DataFrame(id=["A","B","C"]),
		               DataFrame(id=["a","b","c","d"]))

		let d2 = copy(d)
			@test d2.matrix === d.matrix
			@test d2.var !== d.var
			@test d2.var == d.var
			@test d2.obs !== d.obs
			@test d2.obs == d.obs
			@test d2.models !== d.models
			@test d2.models == d.models
		end

		@testset "var=$var obs=$obs matrix=$matrix" for var in (:copy,:keep), obs in (:copy,:keep), matrix in (:copy,:keep)
			d2 = copy(d; var, obs, matrix)

			if var == :copy
				@test d2.var !== d.var
				@test d2.var == d.var
			else
				@test d2.var === d.var
			end

			if obs == :copy
				@test d2.obs !== d.obs
				@test d2.obs == d.obs
			else
				@test d2.obs === d.obs
			end

			if matrix == :copy
				@test d2.matrix !== d.matrix
				@test d2.matrix == d.matrix
			else
				@test d2.matrix === d.matrix
			end

			@test d2.models !== d.models
			@test d2.models == d.models
		end
	end
end
