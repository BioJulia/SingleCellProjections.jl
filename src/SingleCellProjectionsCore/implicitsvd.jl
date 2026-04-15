function svdbyeigen(A; nsv::Integer=3)
	P,N = size(A)
	K = Symmetric(N<=P ? A'A : A*A')
	M = size(K,1)
	F = eigen(K, M-nsv+1:M)
	S = sqrt.(max.(0.,reverse(F.values)))
	denom = max.(1e-100, S) # To avoid NaNs if any singular value is zero

	V = F.vectors[:,end:-1:1]
	N<=P ? SVD(A*V./denom',S,V') : SVD(V,S,V'A./denom)
end

function stabilize_sign!(F::SVD)
	flip = vec(sum(F.U; dims=1).<0)
	F.U[:,flip] .*= -1
	F.Vt[flip,:] .*= -1
	F
end


function implicitsvd(::Type{T}, P, N, A, AT;
                     nsv::Integer=3, subspacedims::Integer=4nsv, niter::Integer=2,
                     stabilize_sign = true,
                     seed = nothing,
                     rng = seed !== nothing ? seed2rng(seed) : Random.default_rng(),
                     verbose = true) where T
	P*N==0 && return SVD(zeros(0,0),zeros(0),zeros(0,0))
	nsv = min(nsv,P,N)
	@assert subspacedims>=nsv
	@assert niter>=0

	local B
	local Q

	if subspacedims>=min(P,N)
		# revert to standard SVD
		# B = convert(Matrix, A*I(N)) # TODO: Add `convert(Matrix, A)` to MatrixExpressions instead and use that.
		B = convert(Matrix, A)
		Q = I
	else
		progress = verbose ? Progress((niter+1)*4; desc="Computing SVD: ") : nothing

		local Zj
		for j=0:niter # TODO: change to 1:niter and alter at call sites? Require niter>=1.
			if j==0
				Ω = randn(rng, T, N, subspacedims)
				# Ω = randn(rng, T, subspacedims, N)' # TESTING
			else
				Ω = Matrix(qr(Zj).Q)
				# Ω = copy(Matrix(qr(Zj).Q)')' # TESTING
			end
			isnothing(progress) || next!(progress)

			# @show typeof(Ω), size(Ω)
			Yj = A*Ω
			# @show typeof(Yj), size(Yj)
			isnothing(progress) || next!(progress)

			Q = Matrix(qr(Yj).Q)
			# Q = copy(Matrix(qr(Yj).Q)')' # TESTING
			isnothing(progress) || next!(progress)

			Zj = AT*Q
			# @show typeof(Zj), size(Zj)
			isnothing(progress) || next!(progress)
		end
		isnothing(progress) || finish!(progress)
		B = convert(Matrix,Zj)'
	end

	F = svdbyeigen(B; nsv=nsv)
	F = SVD(Q*F.U,F.S,F.Vt)
	stabilize_sign && stabilize_sign!(F)
	F
end

_floattype(::Type{T}) where T<:AbstractFloat = T
_floattype(::Type{T}) where T = promote_type(T,Float64)

"""
	implicitsvd(A; nsv=3, subspacedims=4nsv, niter=2, stabilize_sign=true, seed, rng)

Compute the SVD of `A` using Random Subspace SVD. [Halko et al. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions"]

* `nsv` - Number of singular values/vectors to compute
* `subspacedims` - Number of dimensions used for the subspace approximating the action of `A`.
* `niter` - Number of iterations. In each iteration, one multiplication of `A` with a matrix and one multiplication of `A'` with a matrix will be performed.
* `stabilize_sign` - If true, handles the problem that the SVD is only unique up to the sign of each component (for real matrices), by ensuring that the l1 norm of the positive entires for each column in U is larger than the l1 norm of the negative entries.
* `seed` - Use a random seed to init the `rng`. NB: This requires the package `StableRNGs` to be loaded.
* `rng` - Specify a custom RNG.

"""
implicitsvd(A; kwargs...) = implicitsvd(_floattype(eltype(A)), size(A)..., A, A'; kwargs...)
