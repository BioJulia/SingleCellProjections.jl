function transfer_categorical_annotation(adj, annot::AbstractVector{T}) where T
	N1,N2 = size(adj)
	@assert N1 == length(annot)
	outAnnot = Vector{T}(undef, N2)
	outAnnotScore = Vector{Float64}(undef, N2)

	R = rowvals(adj)
	V = nonzeros(adj)

	w = Dict{T,Float64}()

	for j in 1:N2
		empty!(w)
		for k in nzrange(adj, j)
			i = R[k]
			key = annot[i]
			get!(w, key, 0.0)
			w[key] += V[k]
		end
		outAnnotScore[j], outAnnot[j] = findmax(w)
	end

	outAnnot, outAnnotScore
end
