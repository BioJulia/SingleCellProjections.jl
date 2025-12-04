# TODO: We might move the functions defined for these test cases somewhere else, in particular if we split across multiple files

module TestJobs
	function my_rand end
	function my_add end
	function my_sub end
	function my_mul end
	function my_div end

	function dm_rand end
	function dm_add end
	function dm_sub end
	function dm_mul end
	function dm_div end
end


# --- Test jobs ---

my_rand_impl(S, nrow, ncol; seed) = rand(StableRNG(seed), S, nrow, ncol)
my_rand_spec(S, nrow, ncol; seed) = create_spec(my_rand_impl, S, nrow, ncol; seed, __version=v"1.0.0")
TestJobs.my_rand(S, nrow, ncol; seed) = Job(my_rand_spec(S, nrow, ncol; seed))

my_add_impl(a,b) = a .+ b
my_add_spec(a, b) = create_spec(my_add_impl, a, b; __version=v"1.0.0")
TestJobs.my_add(a, b) = Job(my_add_spec(a, b))

# Projectable with the first arg fixed
my_sub_impl(a,b) = a .- b
my_sub(action::Action, a, b) = create_spec(my_sub_impl, a, action(b); __version=v"1.0.0")
my_sub_spec(a, b) = create_spec(Projectable(my_sub), a, b)
TestJobs.my_sub(a, b) = Job(my_sub_spec(a, b))

# Elementwise multiplication
my_mul_impl(a, b) = a .* b
my_mul_spec(a, b) = create_spec(my_mul_impl, a, b; __version=v"1.0.0")
TestJobs.my_mul(a, b) = Job(my_mul_spec(a, b))

# Elementwise division - Projectable with the first arg fixed
my_div_impl(a, b) = a ./ b
my_div(action::Action, a, b) = create_spec(my_div_impl, a, action(b); __version=v"1.0.0")
my_div_spec(a, b) = create_spec(Projectable(my_div), a, b)
TestJobs.my_div(a, b) = Job(my_div_spec(a, b))



# DataMatrix versions
# NB: We don't do anything interesting with obs/var, that is tested elsewhere, this file is just about testing DataMatrixFunction interactions with projections.
dm_rand(::Mat, args...; kwargs...) = my_rand_spec(args...; kwargs...)
dm_rand(::Var, S, nrow, ncol; kwargs...) = SingleCellProjections.prefixed_ids_spec("var_id", "var_", nrow)
dm_rand(::Obs, S, nrow, ncol; kwargs...) = SingleCellProjections.prefixed_ids_spec("obs_id", "obs_", ncol)
TestJobs.dm_rand(args...; kwargs...) = Job(create_spec(DataMatrixFunction(dm_rand), args...; kwargs...))

dm_add(::Mat, a, b) = my_add_spec(get_matrix_spec(a), get_matrix_spec(b))
dm_add(f, a, b) = SingleCellProjections.get_spec(f, a) # Var/Obs
TestJobs.dm_add(a, b) = Job(create_spec(DataMatrixFunction(dm_add), a, b))

dm_sub(::Mat, a, b) = my_sub_spec(get_matrix_spec(a), get_matrix_spec(b))
dm_sub(f, a, b) = SingleCellProjections.get_spec(f, a) # Var/Obs
TestJobs.dm_sub(a, b) = Job(create_spec(DataMatrixFunction(dm_sub), a, b))

dm_mul(::Mat, a, b) = my_mul_spec(get_matrix_spec(a), get_matrix_spec(b))
dm_mul(f, a, b) = SingleCellProjections.get_spec(f, a) # Var/Obs
TestJobs.dm_mul(a, b) = Job(create_spec(DataMatrixFunction(dm_mul), a, b))

dm_div(::Mat, a, b) = my_div_spec(get_matrix_spec(a), get_matrix_spec(b))
dm_div(f, a, b) = SingleCellProjections.get_spec(f, a) # Var/Obs
TestJobs.dm_div(a, b) = Job(create_spec(DataMatrixFunction(dm_div), a, b))


# --- Utilities ---
# Some utilties that make it possible to write more concise `@test` code below
+ʲ(a,b) = TestJobs.my_add(a,b)
-ʲ(a,b) = TestJobs.my_sub(a,b)
*ʲ(a,b) = TestJobs.my_mul(a,b)
/ʲ(a,b) = TestJobs.my_div(a,b)

+ᵈ(a,b) = TestJobs.dm_add(a,b)
-ᵈ(a,b) = TestJobs.dm_sub(a,b)
*ᵈ(a,b) = TestJobs.dm_mul(a,b)
/ᵈ(a,b) = TestJobs.dm_div(a,b)

fwd_spec(job) = forward(job).spec



# --- Tests ---
# TODO: Test much more with forward_once

# Is it worth merging this with the DataMatrixFunction test code below?
# They are almost identical, but merging them will make the test code harder to read.
@testset "Projectables" begin
	P,N = 5,3

	A = TestJobs.my_rand(1:9, P, N; seed=1001)
	A_res = rand(StableRNG(1001), 1:9, P, N)
	Ap = TestJobs.my_rand(1:9, P, N; seed=1002)
	Ap_res = rand(StableRNG(1002), 1:9, P, N)

	B = TestJobs.my_rand(1:9, P, N; seed=1003)
	B_res = rand(StableRNG(1003), 1:9, P, N)
	Bp = TestJobs.my_rand(1:9, P, N; seed=1004)
	Bp_res = rand(StableRNG(1004), 1:9, P, N)

	C = TestJobs.my_rand(1:9, P, N; seed=1005)
	C_res = rand(StableRNG(1005), 1:9, P, N)
	Cp = TestJobs.my_rand(1:9, P, N; seed=1006)
	Cp_res = rand(StableRNG(1006), 1:9, P, N)

	@testset "Basics" begin
		@test allunique([A_res, Ap_res, B_res, Bp_res])
		@test fetch!(A) == A_res
		@test fetch!(Ap) == Ap_res
		@test fetch!(B) == B_res
		@test fetch!(Bp) == Bp_res
		@test fetch!(C) == C_res
		@test fetch!(Cp) == Cp_res
	end

	@testset "Outer replacement" begin
		jp1 = Jobs.project(A, A=>Ap)
		@test fetch!(jp1) == Ap_res
		@test isequal(fwd_spec(jp1), fwd_spec(Ap))
	end

	@testset "my_add" begin
		j1 = TestJobs.my_add(A, B)
		@test fetch!(j1) == A_res+B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1) == Ap_res+Bp_res

		@test isequal(fwd_spec(jp1), fwd_spec(Ap +ʲ Bp))
		@test forward_once(jp1).spec.f == ProjectOnto(my_add_impl)

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced) == Bp_res
	end

	@testset "my_mul" begin
		j1 = TestJobs.my_mul(A, B)
		@test fetch!(j1) == A_res.*B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1) == Ap_res.*Bp_res

		@test isequal(fwd_spec(jp1), fwd_spec(Ap *ʲ Bp))
		@test forward_once(jp1).spec.f == ProjectOnto(my_mul_impl)

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced) == Bp_res
	end

	 # NB: The first argument to my_sub is not affected by projections
	@testset "my_sub" begin
		j1 = TestJobs.my_sub(A, B)
		@test fetch!(j1) == A_res-B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1) == A_res-Bp_res
		@test isequal(fwd_spec(jp1), fwd_spec(A -ʲ Bp))

		jp1fwd = forward_once(jp1)
		@test jp1fwd.spec.f == ProjectOnto(Projectable(my_sub))

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced) == Bp_res
	end

	 # NB: The first argument to my_div is not affected by projections
	@testset "my_div" begin
		j1 = TestJobs.my_div(A, B)
		@test fetch!(j1) == A_res./B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp) # thus, replacnig A=>Ap has now effect
		@test fetch!(jp1) == A_res./Bp_res
		@test isequal(fwd_spec(jp1), fwd_spec(A /ʲ Bp))

		@test forward_once(jp1).spec.f == ProjectOnto(Projectable(my_div))

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced) == Bp_res
	end


	@testset "my_add_div" begin
		j1 = TestJobs.my_add(A, B)
		j2 = TestJobs.my_div(j1, C)

		@test fetch!(j2) == (A_res+B_res) ./ C_res

		jp2 = Jobs.project(j2, C=>Cp)
		@test fetch!(jp2) == (A_res+B_res) ./ Cp_res
		@test isequal(fwd_spec(jp2), fwd_spec((A +ʲ B)/ʲ Cp))
		jp2fwd = forward_once(jp2)
		@test jp2fwd.spec.f == ProjectOnto(Projectable(my_div))
		jp2fwd2 = forward_once(jp2fwd)
		@test jp2fwd2.spec.f == my_div_impl
		# TODO: This will be change so that forwarding is done directly to ProjectOnto(Projectable(my_add))!
		@test jp2fwd2.spec.args[2].f == Preprocess(SingleCellProjections.project)

		jp2b = Jobs.project(j2, B=>Bp) # replacing B=>Bp should have no effect
		@test fetch!(jp2b) == (A_res+B_res) ./ C_res
		@test isequal(fwd_spec(jp2b), fwd_spec((A +ʲ B)/ʲ C))

		jp2c = Jobs.project(j2, j1=>Bp) # replacing j1=>Bp should have no effect
		@test fetch!(jp2c) == (A_res+B_res) ./ C_res
		@test isequal(fwd_spec(jp2c), fwd_spec((A +ʲ B)/ʲ C))
	end

	@testset "my_div_add" begin
		j1 = TestJobs.my_div(A, B)
		j2 = TestJobs.my_add(j1, C)

		@test fetch!(j2) == (A_res./B_res) + C_res

		jp2 = Jobs.project(j2, C=>Cp)
		@test fetch!(jp2) == (A_res./B_res) + Cp_res
		@test isequal(fwd_spec(jp2), fwd_spec(A /ʲ B +ʲ Cp))

		jp2b = Jobs.project(j2, B=>Bp)
		@test fetch!(jp2b) == (A_res./Bp_res) + C_res
		@test isequal(fwd_spec(jp2b), fwd_spec(A /ʲ Bp +ʲ C))

		jp2c = Jobs.project(j2, j1=>Bp)
		@test fetch!(jp2c) == Bp_res + C_res
		@test isequal(fwd_spec(jp2c), fwd_spec(Bp +ʲ C))

		jp2d = Jobs.project(j2, A=>Ap) # replacing A=>Ap should have no effect
		@test fetch!(jp2d) == (A_res./B_res) + C_res
		@test isequal(fwd_spec(jp2d), fwd_spec(A /ʲ B +ʲ C))
	end
end


# Is it worth merging this with the Projectable test code above?
# They are almost identical, but merging them will make the test code harder to read.
@testset "DataMatrixFunction projections" begin
	P,N = 5,3

	A = TestJobs.dm_rand(1:9, P, N; seed=1001)
	A_res = rand(StableRNG(1001), 1:9, P, N)
	Ap = TestJobs.dm_rand(1:9, P, N; seed=1002)
	Ap_res = rand(StableRNG(1002), 1:9, P, N)

	B = TestJobs.dm_rand(1:9, P, N; seed=1003)
	B_res = rand(StableRNG(1003), 1:9, P, N)
	Bp = TestJobs.dm_rand(1:9, P, N; seed=1004)
	Bp_res = rand(StableRNG(1004), 1:9, P, N)

	C = TestJobs.dm_rand(1:9, P, N; seed=1005)
	C_res = rand(StableRNG(1005), 1:9, P, N)
	Cp = TestJobs.dm_rand(1:9, P, N; seed=1006)
	Cp_res = rand(StableRNG(1006), 1:9, P, N)

	var_res = DataFrame("var_id"=>string.("var_", 1:P))
	obs_res = DataFrame("obs_id"=>string.("obs_", 1:N))

	@testset "Basics" begin
		@test allunique([A_res, Ap_res, B_res, Bp_res])
		@test fetch!(A).matrix == A_res
		@test fetch!(Ap).matrix == Ap_res
		@test fetch!(B).matrix == B_res
		@test fetch!(Bp).matrix == Bp_res
		@test fetch!(C).matrix == C_res
		@test fetch!(Cp).matrix == Cp_res

		@test fetch!(A).var == var_res
		@test fetch!(A).obs == obs_res
	end

	@testset "Outer replacement" begin
		jp1 = Jobs.project(A, A=>Ap)
		@test fetch!(jp1).matrix == Ap_res
		@test isequal(fwd_spec(jp1), fwd_spec(Ap))
	end

	@testset "my_add" begin
		j1 = TestJobs.dm_add(A, B)
		@test fetch!(j1).matrix == A_res+B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1).matrix == Ap_res+Bp_res
		@test isequal(fwd_spec(jp1), fwd_spec(Ap +ᵈ Bp))

		@test forward_once(jp1).spec.f == ProjectOnto(DataMatrixFunction(dm_add))

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced).matrix == Bp_res
	end

	@testset "dm_mul" begin
		j1 = TestJobs.dm_mul(A, B)
		@test fetch!(j1).matrix == A_res.*B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1).matrix == Ap_res.*Bp_res
		@test isequal(fwd_spec(jp1), fwd_spec(Ap *ᵈ Bp))

		@test forward_once(jp1).spec.f == ProjectOnto(DataMatrixFunction(dm_mul))

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced).matrix == Bp_res
	end

	# NB: The first argument to dm_sub is not affected by projections
	@testset "dm_sub" begin
		j1 = TestJobs.dm_sub(A, B)
		@test fetch!(j1).matrix == A_res-B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1).matrix == A_res-Bp_res
		@test isequal(fwd_spec(jp1), fwd_spec(A -ᵈ Bp))

		@test forward_once(jp1).spec.f == ProjectOnto(DataMatrixFunction(dm_sub))

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced).matrix == Bp_res
	end

	 # NB: The first argument to dm_div is not affected by projections
	@testset "dm_div" begin
		j1 = TestJobs.dm_div(A, B)
		@test fetch!(j1).matrix == A_res./B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp) # thus, replacnig A=>Ap has now effect
		@test fetch!(jp1).matrix == A_res./Bp_res
		@test isequal(fwd_spec(jp1), fwd_spec(A /ᵈ Bp))

		@test forward_once(jp1).spec.f == ProjectOnto(DataMatrixFunction(dm_div))

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced).matrix == Bp_res
	end


	@testset "dm_add_div" begin
		j1 = TestJobs.dm_add(A, B)
		j2 = TestJobs.dm_div(j1, C)

		@test fetch!(j2).matrix == (A_res+B_res) ./ C_res

		jp2 = Jobs.project(j2, C=>Cp)
		@test fetch!(jp2).matrix == (A_res+B_res) ./ Cp_res
		@test isequal(fwd_spec(jp2), fwd_spec((A +ᵈ B)/ᵈ Cp))
		jp2fwd = forward_once(jp2)
		@test jp2fwd.spec.f == ProjectOnto(DataMatrixFunction(dm_div))

		# NB: Here projectables and DataMatrixFunctions differ a bit
		jp2fwd2 = forward_once(jp2fwd)
		@test jp2fwd2.spec.f == SCPCore.DataMatrix
		jp2fwd3 = Job(jp2fwd2.spec.args[1]) # the matrix arg
		# TODO: This will be change so that forwarding is done directly to ProjectOnto(MatFunction(dm_div))!
		@test jp2fwd3.spec.f == Preprocess(SingleCellProjections.project)
		jp2fwd4 = forward_once(jp2fwd3)
		@test jp2fwd4.spec.f == ProjectOnto(MatFunction(dm_div))

		jp2b = Jobs.project(j2, B=>Bp) # replacing B=>Bp should have no effect
		@test fetch!(jp2b).matrix == (A_res+B_res) ./ C_res
		@test isequal(fwd_spec(jp2b), fwd_spec((A +ᵈ B)/ᵈ C))

		jp2c = Jobs.project(j2, j1=>Bp) # replacing j1=>Bp should have no effect
		@test fetch!(jp2c).matrix == (A_res+B_res) ./ C_res
		@test isequal(fwd_spec(jp2c), fwd_spec((A +ᵈ B)/ᵈ C))
	end

	@testset "dm_div_add" begin
		j1 = TestJobs.dm_div(A, B)
		j2 = TestJobs.dm_add(j1, C)

		@test fetch!(j2).matrix == (A_res./B_res) + C_res

		jp2 = Jobs.project(j2, C=>Cp)
		@test fetch!(jp2).matrix == (A_res./B_res) + Cp_res
		@test isequal(fwd_spec(jp2), fwd_spec(A /ᵈ B +ᵈ Cp))

		jp2b = Jobs.project(j2, B=>Bp)
		@test fetch!(jp2b).matrix == (A_res./Bp_res) + C_res
		@test isequal(fwd_spec(jp2b), fwd_spec(A /ᵈ Bp +ᵈ C))

		jp2c = Jobs.project(j2, j1=>Bp)
		@test fetch!(jp2c).matrix == Bp_res + C_res
		@test isequal(fwd_spec(jp2c), fwd_spec(Bp +ᵈ C))

		jp2d = Jobs.project(j2, A=>Ap) # replacing A=>Ap should have no effect
		@test fetch!(jp2d).matrix == (A_res./B_res) + C_res
		@test isequal(fwd_spec(jp2d), fwd_spec(A /ᵈ B +ᵈ C))
	end
end
