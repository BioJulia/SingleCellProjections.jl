struct DiagMulSubResult
	cost::Float64
	left::Int # should be 1
	split::Int
	right::Int # should be number of factors
	# reverse_mul::Bool # false: AB, true: BA
	child_adj::Tuple{Bool,Bool}
end
DiagMulSubResult() = DiagMulSubResult(Inf,0,0,0,(false,false))


struct DiagMulChainResult
	K::Int # The total number of elements in the chain
	res::Union{DiagMulSubResult, AdjointSparseChainResult}
end
DiagMulChainResult() = DiagMulChainResult(0,AdjointSparseChainResult())
DiagMulChainResult(A::MatrixRef, leaf_ind::Int, K::Int) = DiagMulChainResult(K, AdjointSparseChainResult(A,leaf_ind))


getsubresult(R::DiagMulChainResult, adj) = getsubresult(R.res, adj)

isleaf(S::DiagMulSubResult) = false
getleft(results, S::DiagMulSubResult) = getsubresult(results[S.left,S.split].res, S.child_adj[1])
getright(results, S::DiagMulSubResult) = getsubresult(results[S.split+1,S.right].res, S.child_adj[2])


struct DiagMulOperation
	child_adj::Tuple{Bool,Bool}
end


function add_operations!(co, left, right, S::DiagMulSubResult)
	op = DiagMulOperation(S.child_adj)
	push!(co.operations, (left,right,op,S.cost))
	length(co.operations)
end

function (op::DiagMulOperation)(A,B)
	# @show size(A), size(B)
	U = op.child_adj[1] ? A' : A
	V = op.child_adj[2] ? B' : B

	@assert U isa Adjoint
	@assert !(V isa Adjoint)

	# @show size(U), size(V)
	# @show typeof(U), typeof(V)
	D = compute_diagmul(U,V)
	# @info "DiagMul"
	# @info "$(typeof(U)), $(typeof(V)) -> $(typeof(D))"
	# @info "$(size(U)), $(size(V)) -> $(size(D))"
	D
end


function _diagchain_cost(::Type{AdjDense{TA}}, ::Type{Matrix{TB}}, A::MatrixInfo, B::MatrixInfo)::Float64 where {TA,TB}
	2 * prod(size(A)) # TODO: revise constant
end
function _diagchain_cost(::Type{<:AdjSparse{TA}}, ::Type{Matrix{TB}}, A::MatrixInfo, B::MatrixInfo)::Float64 where {TA,TB}
	2 * prod(size(A)) * A.pnz * SPARSE_TIME_FACTOR # TODO: revise constant
end
function _diagchain_cost(::Type{AdjDense{TA}}, ::Type{<:SparseArrays.AbstractSparseMatrixCSC{TB,<:Any}}, A::MatrixInfo, B::MatrixInfo)::Float64 where {TA,TB}
	2 * prod(size(A)) * B.pnz * SPARSE_TIME_FACTOR # TODO: revise constant
end
function _diagchain_cost(::Type{<:AdjSparse{TA}}, ::Type{<:SparseArrays.AbstractSparseMatrixCSC{TB,<:Any}}, A::MatrixInfo, B::MatrixInfo)::Float64 where {TA,TB}
	2 * prod(size(A))*(A.pnz+B.pnz) * SPARSE_SPARSE_TIME_FACTOR # TODO: revise constant
end


# fallback to infinite cost for everything except diag(AᵀB)
_diagchain_cost(::Type{<:Any}, ::Type{<:Any}, ::MatrixInfo, ::MatrixInfo) = Inf


function _diagchain_cost(A::MatrixInfo, B::MatrixInfo)
	@assert reverse(size(A))==size(B)
	_diagchain_cost(A.type, B.type, A, B)
end

function _diagchain(A::SubResult, B::SubResult, adjA, adjB)
	@assert A.right+1==B.left
	U = adjA ? _adjoint(A.matrixinfo) : A.matrixinfo
	V = adjB ? _adjoint(B.matrixinfo) : B.matrixinfo
	cost = _diagchain_cost(U, V)
	cost += A.cost + B.cost
	DiagMulSubResult(cost, A.left, A.right, B.right, (adjA,adjB))
end

_diagchain(A::AdjointSparseChainResult, B::AdjointSparseChainResult, adjA, adjB) =
	_diagchain(getsubresult(A,adjA), getsubresult(B,adjB), adjA, adjB)


function _diagchain(A::DiagMulChainResult, B::DiagMulChainResult)
	D1 = _diagchain(A.res, B.res, false, false)
	D2 = _diagchain(A.res, B.res, true, false)
	D3 = _diagchain(A.res, B.res, false, true)
	D4 = _diagchain(A.res, B.res, true, true)

	D = D1
	D2.cost < D.cost && (D = D2)
	D3.cost < D.cost && (D = D3)
	D4.cost < D.cost && (D = D4)
	D
end

function chain(A::DiagMulChainResult, B::DiagMulChainResult)
	K = A.K
	@assert B.K == K

	if A.res.adj_res.left==1 && B.res.res.right==K
		S = _diagchain(A, B)
	else
		S = chain(A.res, B.res)
	end
	DiagMulChainResult(K, S)
end


function merge_results!(results::Matrix{DiagMulChainResult}, i, j, R::DiagMulChainResult)
	S = R.res
	S2 = results[i,j].res
	if S isa AdjointSparseChainResult
		S2::AdjointSparseChainResult

		A = min(S2.res, S.res)
		B = min(S2.adj_res, S.adj_res)

		results[i,j] = DiagMulChainResult(R.K, AdjointSparseChainResult(A,B))
	elseif S isa DiagMulSubResult
		S2::DiagMulSubResult
		if S.cost < S2.cost
			results[i,j] = R
		end
	else
		error("Unexpected type")
	end
	nothing
end



function init_chain_element(::Type{DiagMulChainResult}, factors, i, j)
	K = length(factors)
	S = i==1 && j==K ? DiagMulSubResult() : init_chain_element(AdjointSparseChainResult, factors, i, j)
	DiagMulChainResult(K,S)
end



function plan_diag_chain(A::MatrixProduct)
	@assert length(A.factors)>1
	P,N = size(A)
	P==N || throw(DimensionMismatch("Expected a square matrix, got size $((P,N))."))

	results = optimize_chain(DiagMulChainResult, A.factors)
	R = results[1,end].res::DiagMulSubResult
	order = chain_order(Union{DiagMulOperation,AdjointSparseOperation,AdjointSparseCopyOperation}, results, A.factors, R)
	results, order
end

function diag_chain_mul(A::MatrixProduct)
	_, order = plan_diag_chain(A)
	apply_chain(order)
end


# --- printing ---

function printnode_op(io::IO, op::DiagMulOperation)
	print(io, "diag(L")
	op.child_adj[1] && print(io, 'ᵀ')
	print(io, "*R")
	op.child_adj[2] && print(io, 'ᵀ')
	print(io, ')')
end

function print_op_compact(io::IO, co::ChainOrder, left, right, op::DiagMulOperation)
	print(io, "diag(")
	print_compact(io, (co,left))
	op.child_adj[1] && print(io, 'ᵀ')
	print_compact(io, (co,right))
	op.child_adj[2] && print(io, 'ᵀ')
	print(io, ')')
end
