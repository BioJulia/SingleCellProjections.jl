using .SCPCore: check_kwargs

# Call `check_kwargs` with the given keywords, the way a public function would.
_check(allowed; kwargs...) = check_kwargs(kwargs, allowed...)

function run_utils_tests()
	@testset "check_kwargs" begin
		allowed = (:subspacedims, :niter, :stabilize_sign, :rng)

		@test _check(allowed) === nothing
		@test _check(allowed; niter=3) === nothing
		@test _check(allowed; niter=3, rng=1, subspacedims=8, stabilize_sign=false) === nothing
		@test _check(()) === nothing

		@test_throws ArgumentError _check(allowed; nitr=3)
		@test_throws ArgumentError _check(allowed; niter=3, nitr=3) # one good, one bad
		@test_throws ArgumentError _check((); anything=1)
		@test_throws ArgumentError _check(allowed; __version=1)

		# All unknown keywords are reported, not just the first.
		msg = try; _check(allowed; nitr=3, niter=1, wrong=2); catch e; sprint(showerror, e); end
		@test occursin("nitr", msg)
		@test occursin("wrong", msg)
		@test occursin("subspacedims", msg) # lists the accepted ones
	end
end
