function muoversigma(b0::Float64,b1::Float64,theta::Float64,logM::Float64)
	μ = exp(b0+b1*logM)
	σ = sqrt(μ+μ^2/theta)
	μ/σ
end

function muoversigmafactorization(logCellCounts::AbstractVector, logGeneMean::AbstractVector,
                                  b0::Vector{Float64},b1::Vector{Float64},theta::Vector{Float64};
                                  rtol=1e-3,atol=0)
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

	B,xIndUsed,yIndUsed = bilinearapproximation([1,length(xValuesSorted)],[1,length(yValuesSorted)],xValuesSorted,yValuesSorted,f; rtol=rtol, atol=atol)

	A = linearinterpolationmatrix(yValues,yValuesSorted[yIndUsed],true)
	C = linearinterpolationmatrix(xValues,xValuesSorted[xIndUsed],false)

	A,B,C
end


# assumes each column is a cell
function dividebysigma!(X::SparseMatrixCSC{Float64}, logCellCounts::Vector{Float64},
                        β0::Vector{Float64}, β1::Vector{Float64}, θ::Vector{Float64};
                        clip)
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
			V[k] = v
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

dividebysigma(X::SparseMatrixCSC{Float64}, args...; kwargs...) = dividebysigma!(copy(X), args...; kwargs...)
dividebysigma(X::SparseMatrixCSC, args...; kwargs...) = dividebysigma!(convert.(Float64,X), args...; kwargs...)


# assumes each column is a cell
function sctransformsparse(X::SparseMatrixCSC, features, params;
                           transpose = false,
                           feature_id_columns = [:id,:feature_type],
                           cell_ind = 1:size(X,2),
                           clip=sqrt(size(X,2)/30), kwargs...)

	@assert size(X,1)==length(getproperty(features,first(propertynames(features)))) "The number of rows in X and features must match"

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


	logCellCounts = SCTransform.logcellcounts(X)[cell_ind]

	# create new Float64-valued sparse matrix with selected rows/columns

	# X = convert.(Float64, X[feature_ind,cell_ind])

	# A little trick to save memory by avoiding duplicating the rowval and colptr vectors
	# Ideally, this could be done in one step to avoid duplicating nzval too
	X = X[feature_ind,cell_ind]
	X = SparseMatrixCSC(size(X)..., X.colptr, X.rowval, convert.(Float64,X.nzval))


	# compute (approximate) factorization of matrix with elements -μᵢⱼ/σᵢⱼ
	B1,B2,B3 = muoversigmafactorization(logCellCounts, logGeneMean, β0, β1, θ; kwargs...)
	dividebysigma!(X,logCellCounts,β0,β1,θ; clip=clip)

	Y = matrixproduct((:B₁,B1), (:B₂,B2), (:B₃,B3))
	Z = matrixsum((:A,ThreadedSparseMatrixCSC(X)), Y)

	Z = transpose ? Z' : Z
	Z, features[feature_ind,:] # TODO: revise this solution
end
