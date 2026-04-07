function muoversigma(b0::Float64, b1::Float64, theta::Float64, logM::Float64)
	μ = exp(b0+b1*logM)
	σ = sqrt(μ+μ^2/theta)
	μ/σ
end

function muoversigmafactorization(logCellCounts::AbstractVector, logGeneMean::AbstractVector,
                                  b0::AbstractVector{Float64}, b1::AbstractVector{Float64}, theta::AbstractVector{Float64};
                                  rtol = 1e-3, atol = 0.0)
	xValues = logCellCounts
	yValues = logGeneMean

	xValuesSorted = sort(unique(xValues))
	yUniqueInd = unique(i->yValues[i], 1:length(yValues))
	sort!(yUniqueInd, by=i->yValues[i])

	b0    = b0[yUniqueInd]
	b1    = b1[yUniqueInd]
	theta = theta[yUniqueInd]
	yValuesSorted = yValues[yUniqueInd]

	f(xI,yI) = -muoversigma(b0[yI],b1[yI],theta[yI],xValuesSorted[xI])

	B,xIndUsed,yIndUsed = bilinearapproximation([1,length(xValuesSorted)], [1,length(yValuesSorted)], xValuesSorted, yValuesSorted, f; rtol, atol)

	A = linearinterpolationmatrix(yValues, yValuesSorted[yIndUsed], true)
	C = linearinterpolationmatrix(xValues, xValuesSorted[xIndUsed], false)

	A,B,C
end


# assumes each column is a cell
function dividebysigma!(X::SparseMatrixCSC{T}, logCellCounts::AbstractVector{Float64},
                        β0::AbstractVector{Float64}, β1::AbstractVector{Float64}, θ::AbstractVector{Float64};
                        clip) where T
	P,N = size(X)
	@assert clip>0
	@assert P == length(β0)
	@assert P == length(β1)
	@assert P == length(θ)

	# input
	R = rowvals(X)
	V = nonzeros(X)

	for j=1:N
		for k in nzrange(X,j)
			i = R[k]
			μ = exp(β0[i]+β1[i]*logCellCounts[j])
			σ = sqrt(μ+μ^2/θ[i])
			v = clamp(V[k]/σ, -clip+μ/σ, clip+μ/σ) # ensure -clip <= x/σ-μ/σ <= clip
			V[k] = convert(T, v)
		end
	end
	X
end


# assumes each column is a gene
# function dividebysigma!(X::SparseMatrixCSC{Float64}, logCellCounts::Vector{Float64},
#                         β0::Vector{Float64}, β1::Vector{Float64}, θ::Vector{Float64};
#                         clip)
# 	N,P = size(X)
# 	@assert clip>0
# 	@assert P == length(β0)
# 	@assert P == length(β1)
# 	@assert P == length(θ)

# 	# input
# 	R = rowvals(X)
# 	V = nonzeros(X)

# 	for j=1:P
# 		for k in nzrange(X,j)
# 			i = R[k]
# 			μ = exp(β0[j]+β1[j]*logCellCounts[i])
# 			σ = sqrt(μ+μ^2/θ[j])
# 			v = clamp(V[k]/σ, -clip+μ/σ, clip+μ/σ) # ensure -clip <= x/σ-μ/σ <= clip
# 			V[k] = v
# 		end
# 	end
# 	X
# end

# dividebysigma(X::SparseMatrixCSC{Float64}, args...; kwargs...) = dividebysigma!(copy(X), args...; kwargs...)
# dividebysigma(X::SparseMatrixCSC, args...; kwargs...) = dividebysigma!(convert.(Float64,X), args...; kwargs...)


# assumes each column is a cell
function sctransformsparse(::Type{T}, X::SparseMatrixCSC, features, params;
                           transpose = false,
                           feature_id_columns = [:id,:feature_type],
                           feature_mask,
                           cell_ind = 1:size(X,2),
                           clip=sqrt(size(X,2)/30), kwargs...) where T

	@assert size(X,1)==length(getproperty(features,first(propertynames(features)))) "The number of rows in X and features must match"

	feature_mask = convert(BitVector, feature_mask)

	β0 = params.beta0
	β1 = params.beta1
	θ  = params.theta
	logGeneMean = params.logGeneMean

	# Use features to figure out which rows in params match which rows in X
	# TODO: cleanup code
	param_ids = getproperty(params, feature_id_columns[1])
	feature_ids = getproperty(features, feature_id_columns[1])
	for col in feature_id_columns[2:end]
		param_ids = string.(param_ids, "__<sep>__", getproperty(params, col))
		feature_ids = string.(feature_ids, "__<sep>__", getproperty(features, col))
	end
	feature_ind = indexin(param_ids, feature_ids)
	any(isnothing, feature_ind) && throw(DomainError("Feature ids in `params` does not match ids in `features`."))
	feature_ind = Int.(feature_ind) # get rid of Nothing in eltype


	logCellCounts = SCTransform.logcellcounts(X, feature_mask)[cell_ind]

	# create new Float64-valued sparse matrix with selected rows/columns

	# X = convert.(Float64, X[feature_ind,cell_ind])

	# A little trick to save memory by avoiding duplicating the rowval and colptr vectors
	# Ideally, this could be done in one step to avoid duplicating nzval too
	X = X[feature_ind,cell_ind]
	X = SparseMatrixCSC(size(X)..., X.colptr, X.rowval, convert.(T,X.nzval))


	# compute (approximate) factorization of matrix with elements -μᵢⱼ/σᵢⱼ
	B1,B2,B3 = muoversigmafactorization(logCellCounts, logGeneMean, β0, β1, θ; kwargs...)
	dividebysigma!(X,logCellCounts,β0,β1,θ; clip=clip)

	Y = matrixproduct((:B₁,B1), (:B₂,B2), (:B₃,B3))
	Z = matrixsum((:A,X), Y)

	Z = transpose ? Z' : Z
	Z, features[feature_ind,:] # TODO: revise this solution
end
sctransformsparse(X::SparseMatrixCSC, args...; kwargs...) =
	sctransformsparse(Float64, X, args...; kwargs...)




function _sctransform_sparse(::Type{T}, X::SparseMatrixCSC, feature_mask, log_cell_counts, β0, β1, θ; clip) where T
	# A little trick to save memory by avoiding duplicating the rowval and colptr vectors
	# Ideally, this could be done in one step to avoid duplicating nzval too
	X = X[feature_mask,:]
	X = SparseMatrixCSC(size(X)..., X.colptr, X.rowval, convert.(T,X.nzval))
	dividebysigma!(X, log_cell_counts, β0, β1, θ; clip=clip)
end


function _sctransform_sparse(::Type{T}, A::Blocks{SparseMatrixCSC{Tv,Ti}}, feature_mask, log_cell_counts, β0, β1, θ; clip) where {T,Tv,Ti}
	row_ranges = get_row_ranges(A)
	col_ranges = get_col_ranges(A)

	param_ranges = UnitRange{Int}[]
	let s = 1
		for r in row_ranges
			next = s + count(feature_mask[r])
			push!(param_ranges, s:next-1)
			s = next
		end
	end

	blocks = Matrix{SparseMatrixCSC{T,Ti}}(undef, size(A.blocks))
	for i in 1:size(A.blocks, 1) # TODO: thread?
		fm_view = @view feature_mask[row_ranges[i]]

		# NB: β0, β1 and θ corresponds to already masked variables. So they do not match row_ranges.
		β0_view = @view β0[param_ranges[i]]
		β1_view = @view β1[param_ranges[i]]
		θ_view = @view θ[param_ranges[i]]

		for j in 1:size(A.blocks, 2)
			lcc_view = @view log_cell_counts[col_ranges[j]]
			blocks[i,j] = _sctransform_sparse(T, A.blocks[i,j], fm_view, lcc_view, β0_view, β1_view, θ_view; clip)
		end
	end

	# Remove blocks with height 0
	if any(iszero, size.(@view(blocks[:,1]),1))
		blocks = blocks[.!iszero.(size.(@view(blocks[:,1]),1)), :]
	end

	# TODO: Redo blocking to get back original block sizes?

	Blocks(blocks)
end




# # assumes each column is a cell
# function sctransformsparse2(::Type{T}, X, params, feature_ind, log_cell_counts;
#                             # cell_ind = 1:size(X,2),
#                             clip=sqrt(size(X,2)/30),
#                             kwargs...) where T
# 	# @assert size(X,1)==length(getproperty(features,first(propertynames(features)))) "The number of rows in X and features must match"
# 	@assert size(params,1) == length(feature_ind)
# 	@assert length(log_cell_counts) == size(X,2)

# 	feature_mask = falses(size(X,1))
# 	feature_mask[feature_ind] .= true

# 	β0 = params.beta0
# 	β1 = params.beta1
# 	θ  = params.theta
# 	logGeneMean = params.logGeneMean

# 	# logCellCounts = SCTransform.logcellcounts(X, feature_mask)[cell_ind]

# 	# create new Float64-valued sparse matrix with selected rows/columns

# 	# X = convert.(Float64, X[feature_ind,cell_ind])

# 	# A little trick to save memory by avoiding duplicating the rowval and colptr vectors
# 	# Ideally, this could be done in one step to avoid duplicating nzval too
# 	# X = X[feature_ind,:]
# 	# X = SparseMatrixCSC(size(X)..., X.colptr, X.rowval, convert.(T,X.nzval))
# 	# dividebysigma!(X, log_cell_counts, β0, β1, θ; clip)

# 	X = _sctransform_sparse(T, X, feature_mask, log_cell_counts, β0, β1, θ; clip)

# 	# compute (approximate) factorization of matrix with elements -μᵢⱼ/σᵢⱼ
# 	B1,B2,B3 = muoversigmafactorization(log_cell_counts, logGeneMean, β0, β1, θ; kwargs...)

# 	Y = matrixproduct((:B₁,B1), (:B₂,B2), (:B₃,B3))
# 	Z = matrixsum((:A,X), Y)
# 	Z
# end
# sctransformsparse2(X, params, feature_ind, log_cell_counts; kwargs...) =
# 	sctransformsparse2(Float64, X, params, feature_ind, log_cell_counts; kwargs...)


# assumes each column is a cell
function sctransformsparse_a(::Type{T}, X, params, feature_ind, log_cell_counts;
                             clip=sqrt(size(X,2)/30)) where T
	@assert size(params,1) == length(feature_ind)
	@assert length(log_cell_counts) == size(X,2)

	feature_mask = falses(size(X,1))
	feature_mask[feature_ind] .= true

	β0 = params.beta0
	β1 = params.beta1
	θ  = params.theta
	logGeneMean = params.logGeneMean

	X = _sctransform_sparse(T, X, feature_mask, log_cell_counts, β0, β1, θ; clip)
end
sctransformsparse_a(X, params, feature_ind, log_cell_counts; kwargs...) =
	sctransformsparse_a(Float64, X, params, feature_ind, log_cell_counts; kwargs...)


function sctransformsparse_b(params, log_cell_counts; kwargs...)
	β0 = params.beta0
	β1 = params.beta1
	θ  = params.theta
	logGeneMean = params.logGeneMean

	# compute (approximate) factorization of matrix with elements -μᵢⱼ/σᵢⱼ
	B1,B2,B3 = muoversigmafactorization(log_cell_counts, logGeneMean, β0, β1, θ; kwargs...)

	# TEMP, we should somehow match the block sizes from sctransformsparse_a instead
	# B3 = blockify(B3; row_block_size=typemax(Int), col_block_size=1024)

	matrixproduct((:B₁,B1), (:B₂,B2), (:B₃,B3))
end
