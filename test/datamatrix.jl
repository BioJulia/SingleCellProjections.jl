@testset "DataMatrix" begin
	@testset "Basic" begin
		d = DataMatrix()
		@test size(d) == (0,0)
		@test d.var == DataFrame(;id=String[]) == d.obs
		@test d.var_id_cols == ["id"] == d.obs_id_cols
		@test string(d) == "0×0 Matrix{Float64}"

		d = DataMatrix([11 21 31 41; 12 22 32 42; 13 23 33 43],
		               DataFrame(id=["A","B","C"], id2=["Z","Y","X"], v1=["A","A","B"], v2=["1","2","2"]),
		               DataFrame(id=["a","b","c","d"], o1=["A","A","B","B"], o2=["1","2","1","2"]))
		@test size(d) == (3,4)
		@test d.matrix == [11 21 31 41; 12 22 32 42; 13 23 33 43]
		@test d.var == DataFrame(id=["A","B","C"], id2=["Z","Y","X"], v1=["A","A","B"], v2=["1","2","2"])
		@test d.obs == DataFrame(id=["a","b","c","d"], o1=["A","A","B","B"], o2=["1","2","1","2"])
		@test d.var_id_cols == ["id"] == d.obs_id_cols
		@test string(d) == "3×4 Matrix{$Int}"
		test_show(d; matrix="Matrix{$Int}", var=["id","id2","v1","v2"], obs=["id","o1","o2"], models=r"^$")

		set_var_id_cols!(d, ["id2"])
		@test d.var_id_cols == ["id2"]
		@test_throws ArgumentError set_var_id_cols!(d, ["nonexisting"])
		@test_throws ErrorException set_var_id_cols!(d, ["v1"])
		@test_throws ErrorException set_var_id_cols!(d, ["v2"])
		set_var_id_cols!(d, ["v1", "v2"])
		@test d.var_id_cols == ["v1", "v2"]
		set_var_id_cols!(d, ["id"])
		@test d.var_id_cols == ["id"]

		@test_throws ArgumentError set_obs_id_cols!(d, ["nonexisting"])
		@test_throws ErrorException set_obs_id_cols!(d, ["o1"])
		@test_throws ErrorException set_obs_id_cols!(d, ["o2"])
		set_obs_id_cols!(d, ["o1", "o2"])
		@test d.obs_id_cols == ["o1", "o2"]
		set_obs_id_cols!(d, ["id"])
		@test d.obs_id_cols == ["id"]

		push!(d.var_id_cols, "vbad")
		push!(d.obs_id_cols, "obad")
		test_show(d; matrix="Matrix{$Int}", var=["id","id2","v1","v2","vbad"], obs=["id","o1","o2","obad"], models=r"^$")

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
			@test d2.var_id_cols !== d.var_id_cols
			@test d2.var_id_cols == d.var_id_cols
			@test d2.obs_id_cols !== d.obs_id_cols
			@test d2.obs_id_cols == d.obs_id_cols
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

			@test d2.var_id_cols !== d.var_id_cols
			@test d2.var_id_cols == d.var_id_cols
			@test d2.obs_id_cols !== d.obs_id_cols
			@test d2.obs_id_cols == d.obs_id_cols
			@test d2.models !== d.models
			@test d2.models == d.models
		end
	end
end
