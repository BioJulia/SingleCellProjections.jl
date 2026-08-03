# NB: This function is very similar to embed_points - can we share some code?
function transfer_categorical_annotation(f, base_data, base_annot::AbstractVector{T}, data, indices) where T
	base_N = size(base_data,2)
	N = size(data,2)
	@assert size(base_data,1) == size(data,1)
	@assert size(base_data,2) == length(base_annot)
	@assert N == size(indices,2)

	out_annot = Vector{T}(undef, N)
	out_score = Vector{Float64}(undef, N)

	scratch = TaskLocalValue{Dict{T,Float64}}(() -> Dict{T,Float64}())
	# found = Dict{T,Float64}()

	# for j in 1:N # TODO: Thread
	# tforeach(1:N) do j # Configure scheduler?
	tforeach(1:N; scheduler=:greedy, chunking=true, minchunksize=128) do j # TODO: Revisit parameters
		found = scratch[]
		empty!(found)

		total_weight = 0.0
		for base_j in @view(indices[:,j])
			d2 = mapreduce((a,b)->abs2(a-b), +, @view(data[:,j]), @view(base_data[:,base_j])) # TODO: Maybe write optimized function for this with @inbounds and @simd?
			w = f(d2)

			key = base_annot[base_j]
			get!(found, key, 0.0)
			found[key] += w
			total_weight += w
		end

		best_w, best_value = findmax(found)
		out_annot[j] = best_value
		out_score[j] = best_w / total_weight
	end

	out_annot, out_score
end
