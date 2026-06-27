# # Simpler case when we do not need to reorder
# function _logtransform_simple(::Type{T}, X::SparseMatrixCSC{Tv,Ti}; scale_factor) where {T,Tv,Ti}
# 	P,N = size(X)
# 	s = max.(1, sum(X; dims=1))
# 	nf = scale_factor ./ s

# 	# log( 1 + c*f/s)
# 	nzval = nonzeros(X)
# 	nzval_out = zeros(T, nnz(X))

# 	# TODO: parallelize
# 	for j in 1:N
# 		irange = nzrange(X,j)
# 		# nzval_out[irange] .= log2.(1 .+ (@view nzval[irange]) .* nf[j])
# 		nzval_out[irange] .= convert.(T, log2.(1 .+ (@view nzval[irange]) .* nf[j]))
# 	end

# 	# A = SparseMatrixCSC(P, N, copy(X.colptr), copy(X.rowval), nzval_out)
# 	# MatrixRef(:A=>A)
# 	SparseMatrixCSC(P, N, copy(X.colptr), copy(X.rowval), nzval_out)
# end

# # This is slightly complicated to avoid using large temporary arrays
# function _logtransform_reorder(::Type{T}, X::SparseMatrixCSC{Tv,Ti}; scale_factor, var_ind) where {T,Tv,Ti}
# 	P,N = size(X) # original size
# 	P_out = length(var_ind)
# 	ind_map = something.(indexin(1:P, var_ind), 0)

# 	R = rowvals(X)
# 	V = nonzeros(X)

# 	# colptr = similar(X.colptr)
# 	s = zeros(Tv, N) # columns sums
# 	c = zeros(Ti, N+1) # number of elements in previous column! Needed for colptr below.

# 	# TODO: parallelize over columns
# 	for j in 1:N
# 		for k in nzrange(X,j)
# 			i_out = ind_map[R[k]]
# 			i_out == 0 && continue

# 			s[j] += V[k]
# 			c[j+1] += 1
# 		end
# 	end

# 	s .= max.(1, s) # avoid div by zero
# 	nf = scale_factor ./ s

# 	c[1] = 1 # colptr always starts at 1
# 	# colptr = cumsum(c)
# 	colptr = accumulate(+, c) # cumsum changes eltype which we don't want

# 	nnz_out = colptr[end]-1
# 	nzval_out = zeros(T, nnz_out)
# 	rowval_out = zeros(Ti, nnz_out)

# 	do_sort = !issorted(var_ind)

# 	# TODO: parallelize over columns
# 	for j in 1:N
# 		rng = nzrange(X,j)

# 		is_out = ind_map[R[rng]]
# 		vs = V[rng]

# 		if do_sort
# 			perm = sortperm(is_out)
# 			start = something(findfirst(>(0), perm), length(perm)+1) # Maybe use searchsortedfirst? Benchmark to decide.
# 			perm2 = @view perm[start,end]
# 			is_out = is_out[perm2]
# 			vs = vs[perm2]
# 		else
# 			mask = is_out .!= 0
# 			is_out = is_out[mask]
# 			vs = vs[mask]
# 		end

# 		rng_out = colptr[j]:colptr[j+1]-1
# 		rowval_out[rng_out] .= is_out
# 		nzval_out[rng_out] .= convert.(T, log2.(1 .+ vs.*nf[j]))
# 	end

# 	# A = SparseMatrixCSC(P_out, N, colptr, rowval_out, nzval_out)
# 	# MatrixRef(:A=>A)
# 	SparseMatrixCSC(P_out, N, colptr, rowval_out, nzval_out)
# end


# function _logtransform_simple(::Type{T}, X::Matrix; scale_factor) where T
# 	s = max.(1, sum(X; dims=1))
# 	nf = scale_factor ./ s
# 	# A = convert.(T, log2.(1 .+ X.*nf))
# 	# MatrixRef(:A=>A)
# 	convert.(T, log2.(1 .+ X.*nf))
# end

# # This can be optimized to use less memory, but if using dense matrices, that might not be a priority
# _logtransform_reorder(::Type{T}, X::Matrix; scale_factor, var_ind) where T =
# 	_logtransform_simple(T, X[var_ind,:]; scale_factor)

# function logtransform_matrix(::Type{T}, X; scale_factor, var_ind=:) where T
# 	if var_ind == Colon() || var_ind == 1:size(X,1)
# 		_logtransform_simple(T, X; scale_factor)
# 	else
# 		P = size(X,1)
# 		@assert all(i->1<=i<=P, var_ind)
# 		@assert allunique(var_ind) # Hmm. This requires Julia 1.11. Are we ready for that?
# 		_logtransform_reorder(T, X; scale_factor, var_ind)
# 	end
# end
# logtransform_matrix(X; kwargs...) = logtransform_matrix(Float64, X; kwargs...)




function _logtransform_matrix(::Type{T}, X::SparseMatrixCSC{Tv,Ti}, s) where {T,Tv,Ti}
	P,N = size(X)

	nzval = nonzeros(X)
	nzval_out = zeros(T, nnz(X))

	for j in 1:N
		irange = nzrange(X,j)
		nzval_out[irange] .= convert.(T, log2.(1 .+ (@view nzval[irange]) .* s[j]))
	end
	SparseMatrixCSC(P, N, X.colptr, X.rowval, nzval_out) # Since we assume immutable data, we can reuse colptr/rowval
end

_logtransform_matrix(::Type{T}, X::Matrix, s) where T =
	convert.(T, log2.(1 .+ X.*reshape(s,1,:)))



function _logtransform_matrix(::Type{T}, A::Blocks{T2}, s) where {T,T2}
	col_ranges = get_col_ranges(A)

	blocks = tmap(eachindex(A.blocks)) do linear_ind
		i,j = Tuple(CartesianIndices(A.blocks)[linear_ind])
		s_view = @view s[col_ranges[j]]
		_logtransform_matrix(T, A.blocks[i,j], s_view)
	end

	blocks = reshape(blocks, size(A.blocks))
	blocks = convert(Matrix, blocks) # This is a no-op in Julia 1.11+
	Blocks(blocks)
end




# # TODO: Put these as more generic utils related to Blocks (see colsum!/colsum)
# _var_sum!(out, X::AbstractMatrix) = sum!(out, X; init=false)
# function _var_sum!(out, A::Blocks{T}) where T
# 	@assert size(out,1) == 1
# 	@assert size(out,2) == size(A,2)

# 	col_ranges = get_col_ranges(A)

# 	tforeach(1:size(A.blocks, 2)) do j # Configure scheduler?
# 		out_view = @view out[:, col_ranges[j]]
# 		for i in 1:size(A.blocks, 1)
# 			_var_sum!(out_view, A.blocks[i,j])
# 		end
# 	end
# 	out
# end


function logtransform_matrix(::Type{T}, X; scale_factor) where T
	cs = colsum(X)
	obs_scale_factor = scale_factor ./ max.(1, cs)
	_logtransform_matrix(T, X, obs_scale_factor)
end

logtransform_matrix(X; kwargs...) = logtransform_matrix(Float64, X; kwargs...)



# function sctransform_matrix(::Type{T}, X, params::DataFrame, var_ind; kwargs...) where T
# 	sctransformsparse2(T, X, params, var_ind; kwargs...)
# end
# function sctransform_matrix(::Type{T}, X, params::DataFrame, var_ind, log_cell_counts; kwargs...) where T
# 	sctransformsparse2(T, X, params, var_ind, log_cell_counts; kwargs...)
# end
# sctransform_matrix(X; kwargs...) = sctransform_matrix(Float64, X; kwargs...)
