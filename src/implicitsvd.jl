function svdbyeigen(A; nsv::Integer=3)
	P,N = size(A)
	K = Symmetric(N<=P ? A'A : A*A')
	M = size(K,1)
	F = eigen(K, M-nsv+1:M)
	S = sqrt.(max.(0.,reverse(F.values)))

	V = F.vectors[:,end:-1:1]
	N<=P ? SVD(A*V./S',S,V') : SVD(V,S,V'A./S)
end

"""
	implicitsvd([T], A; nsv=3, subspacedims=4nsv, niter=2)
"""
function implicitsvd(::Type{T}, P, N, A, AT; nsv::Integer=3, subspacedims::Integer=4nsv, niter::Integer=2,
                     rng = Random.default_rng()) where T
	P*N==0 && return SVD(zeros(0,0),zeros(0),zeros(0,0))
	nsv = min(nsv,P,N)
	@assert subspacedims>=nsv
	@assert niter>=0

	local B
	local Q

	if subspacedims>=min(P,N)
		# revert to standard SVD.
		B = convert(Matrix, A*I(N))
		Q = I
	else
		local Zj
		for j=0:niter # TODO: change to 1:niter and alter at call sites? Require niter>=1.
			if j==0
				Ω = randn(rng, T, N, subspacedims)
			else
				Ω = Matrix(qr(Zj).Q)
			end
			Yj = A*Ω
			Q = Matrix(qr(Yj).Q)
			Zj = AT*Q
		end
		B = convert(Matrix,Zj)'
	end

	F = svdbyeigen(B; nsv=nsv)
	SVD(Q*F.U,F.S,F.Vt)
end

_floattype(::Type{T}) where T<:AbstractFloat = T
_floattype(::Type{T}) where T = promote_type(T,Float64)

implicitsvd(A; kwargs...) = implicitsvd(_floattype(eltype(A)), size(A)..., A, A'; kwargs...)
