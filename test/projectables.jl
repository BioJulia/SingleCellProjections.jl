# TODO: We might move the functions defined for these test cases somewhere else, in particular if we split across multiple files

module TestJobs
	function my_rand end
	function my_add end
	function my_sub end
	function my_mul end
	function my_div end
end

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

# TODO: Test much more with forward_once
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
		@test isequal(forward(jp1).spec, forward(Ap).spec)
	end

	@testset "my_add" begin
		j1 = TestJobs.my_add(A, B)
		@test fetch!(j1) == A_res+B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1) == Ap_res+Bp_res

		@test isequal(forward(jp1).spec, forward(TestJobs.my_add(Ap, Bp)).spec)
		@test forward_once(jp1).spec.f == ProjectOnto(my_add_impl)

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced) == Bp_res
	end

	 # NB: The first argument to my_sub is not affected by projections
	@testset "my_sub" begin
		j1 = TestJobs.my_sub(A, B)
		@test fetch!(j1) == A_res-B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1) == A_res-Bp_res

		@test isequal(forward(jp1).spec, forward(TestJobs.my_sub(A, Bp)).spec)
		@test forward_once(jp1).spec.f == ProjectOnto(Projectable(my_sub))

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced) == Bp_res
	end

	@testset "my_mul" begin
		j1 = TestJobs.my_mul(A, B)
		@test fetch!(j1) == A_res.*B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp)
		@test fetch!(jp1) == Ap_res.*Bp_res

		@test isequal(forward(jp1).spec, forward(TestJobs.my_mul(Ap, Bp)).spec)
		@test forward_once(jp1).spec.f == ProjectOnto(my_mul_impl)

		replaced = Jobs.project(j1, j1=>Bp)
		@test fetch!(replaced) == Bp_res
	end

	 # NB: The first argument to my_div is not affected by projections
	@testset "my_div" begin
		j1 = TestJobs.my_div(A, B)
		@test fetch!(j1) == A_res./B_res

		jp1 = Jobs.project(j1, A=>Ap, B=>Bp) # thus, replacnig A=>Ap has now effect
		@test fetch!(jp1) == A_res./Bp_res

		@test isequal(forward(jp1).spec, forward(TestJobs.my_div(A, Bp)).spec)
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

		jp2b = Jobs.project(j2, B=>Bp) # replacing B=>Bp should have no effect
		@test fetch!(jp2b) == (A_res+B_res) ./ C_res

		jp2c = Jobs.project(j2, j1=>Bp) # replacing j1=>Bp should have no effect
		@test fetch!(jp2c) == (A_res+B_res) ./ C_res
	end

	@testset "my_div_add" begin
		j1 = TestJobs.my_div(A, B)
		j2 = TestJobs.my_add(j1, C)

		@test fetch!(j2) == (A_res./B_res) + C_res

		jp2 = Jobs.project(j2, C=>Cp)
		@test fetch!(jp2) == (A_res./B_res) + Cp_res

		jp2b = Jobs.project(j2, B=>Bp)
		@test fetch!(jp2b) == (A_res./Bp_res) + C_res

		jp2c = Jobs.project(j2, j1=>Bp)
		@test fetch!(jp2c) == Bp_res + C_res

		jp2d = Jobs.project(j2, A=>Ap) # replacing A=>Ap should have no effect
		@test fetch!(jp2d) == (A_res./B_res) + C_res
	end



end
