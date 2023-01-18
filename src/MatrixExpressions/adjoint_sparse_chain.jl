# TODO: These should be established by some (very rough) benchmarking
const SPARSE_TIME_FACTOR = 3
const SPARSE_SPARSE_TIME_FACTOR = 6

const MAKE_DENSE_PNZ = 0.4

_pnz(A) = -1.0
_pnz(A::SparseArrays.AbstractSparseMatrixCSC) = nnz(A)/length(A)
_pnz(A::Adjoint) = _pnz(A')


struct MatrixInfo
	sz::Tuple{Int,Int}
	type::DataType
	pnz::Float64 # set to -1.0 for dense
end
MatrixInfo(sz,type) = MatrixInfo(sz,type,-1.0)
MatrixInfo() = MatrixInfo((0,0),Nothing)
MatrixInfo(A) = MatrixInfo(size(A), typeof(A), _pnz(A))


Base.size(M::MatrixInfo) = M.sz
Base.size(M::MatrixInfo, ind) = ind>2 ? 1 : M.sz[ind]

_is_adjoint(T::Type) = T <: Adjoint
_is_adjoint(M::MatrixInfo) = _is_adjoint(M.type)


# is there a nicer way to do this?
(_remove_adjoint(::Type{<:T}) where T<:Adjoint{<:Any,S} where S) = S
_remove_adjoint(T::UnionAll) = _remove_adjoint(T.body)

_add_adjoint(::Type{T}) where T<:AbstractMatrix{S} where S = Adjoint{S,T}
_add_adjoint(T::Type) = Adjoint{<:Any,T}

_adjoint_type(T) = _is_adjoint(T) ? _remove_adjoint(T) : _add_adjoint(T)
_strip_adjoint(T::Type) = _is_adjoint(T) ? _remove_adjoint(T) : T

_adjoint(M::MatrixInfo) = MatrixInfo(reverse(M.sz), _adjoint_type(M.type), M.pnz)



_is_dense(T::Type) = _strip_adjoint(T) <: Matrix
_is_dense(M::MatrixInfo) = _is_dense(M.type)

_is_sparse(T::Type) = _strip_adjoint(T) <: SparseArrays.AbstractSparseMatrixCSC
_is_sparse(M::MatrixInfo) = _is_sparse(M.type)



struct SubResult # TODO: Rename
	cost::Float64
	pre_cost::Float64 # The cost before converting to dense or making copy of adjoint
	matrixinfo::MatrixInfo
	# how it was constructed
	copy_adj::Bool # This was created by copying the adjoint of the operation described below
	make_dense::Bool
	left::Int
	split::Int # 0 used for leaves
	right::Int
	reverse_mul::Bool # false: AB, true: BA
	child_adj::Tuple{Bool,Bool}
end
SubResult() = SubResult(Inf, Inf, MatrixInfo(), false, false, 0, -1, 0, false, (false,false))
SubResult(cost, pre_cost, matrixinfo, leaf_ind) = SubResult(cost, pre_cost, matrixinfo, false, false, leaf_ind, 0, leaf_ind, false, (false,false))
SubResult(A,leaf_ind) = SubResult(0.0, 0.0, MatrixInfo(A), leaf_ind)

Base.size(R::SubResult) = size(R.matrixinfo)
Base.size(R::SubResult, ind) = size(R.matrixinfo, ind)

# Total ordering for comparable SubResults
_to_tuple(R::SubResult) = (R.cost,R.pre_cost,R.make_dense,R.copy_adj,R.left,R.split,R.right,R.reverse_mul,R.child_adj)
Base.isless(A::SubResult,B::SubResult) = isless(_to_tuple(A),_to_tuple(B))




struct AdjointSparseChainResult
	res::SubResult
	adj_res::SubResult
end
AdjointSparseChainResult() = AdjointSparseChainResult(SubResult(),SubResult())

function AdjointSparseChainResult(A, leaf_ind::Int)
	AdjointSparseChainResult(_check_make_dense(SubResult(A,leaf_ind)),
	                         _check_make_dense(_copy_adjoint(SubResult(A,leaf_ind))))
end
AdjointSparseChainResult(A::MatrixRef, leaf_ind::Int) = AdjointSparseChainResult(A.matrix, leaf_ind)

getsubresult(R::AdjointSparseChainResult, adj) = adj ? R.adj_res : R.res



isleaf(S::SubResult) = S.split==0
getleafind(S::SubResult) = S.left
_getchild(results, S::SubResult, first::Bool) =
	first ? results[S.left,S.split] : results[S.split+1,S.right]
getleft(results, S::SubResult) = getsubresult(_getchild(results, S, !S.reverse_mul), S.child_adj[1] != S.reverse_mul)
getright(results, S::SubResult) = getsubresult(_getchild(results, S, S.reverse_mul), S.child_adj[2] != S.reverse_mul)




struct AdjointSparseOperation
	child_adj::Tuple{Bool,Bool}
end
struct AdjointSparseCopyOperation
	# adj::Bool
	adj_in::Bool
	adj_out::Bool
	make_dense::Bool
end

function add_operations!(co, left, right, S::SubResult)
	if right != 0 # this isn't a leaf
		op = AdjointSparseOperation(S.child_adj)
		push!(co.operations, (left,right,op, S.pre_cost))
		left = length(co.operations)
		right = 0
	end

	if S.copy_adj || S.make_dense
		# @info "add_op"
		# @show S.copy_adj
		# @show S.make_dense
		# @show S.matrixinfo.type

		adj = _is_adjoint(S.matrixinfo)
		if S.copy_adj
			op = AdjointSparseCopyOperation(!adj, adj, S.make_dense)
		else
			op = AdjointSparseCopyOperation(adj, adj, S.make_dense)
		end

		push!(co.operations, (left,right,op,S.cost))
		left = length(co.operations)
	end

	left
end



function (op::AdjointSparseOperation)(A,B)
	# @show size(A), size(B)
	U = op.child_adj[1] ? A' : A
	V = op.child_adj[2] ? B' : B

	# @show size(U), size(V)
	# @show typeof(U), typeof(V)
	@assert !issparse(U) || !issparse(V) || (!(U isa Adjoint) && !(V isa Adjoint)) "Internal error, got AᵀB, ABᵀ or AᵀBᵀ where A and B are sparse."

	# U*V
	X = U*V
	# @info "Mul"
	# @info "$(typeof(U)),$(typeof(V)) -> $(typeof(X))"
	# @info "$(size(U)),$(size(V)) -> $(size(X))"
	X
end


function (op::AdjointSparseCopyOperation)(A)
	@assert op.make_dense || (A isa Adjoint)!=op.adj_in || (A isa Diagonal)

	A isa Diagonal && !op.make_dense && return A # Symmetric, no need to make a copy

	# @info "copy()"
	# @show op.adj_in
	# @show op.adj_out
	# @show size(A)
	# @show typeof(A)

	if op.adj_in
		A = A'
	end

	X = op.make_dense ? convert(Matrix,A) : copy(A)

	if op.adj_out
		X = X'
	end

	# @show size(X)
	# @show typeof(X)
	X
end



const AdjDense = Adjoint{T,Matrix{T}} where T
const AdjSparse = Adjoint{T,<:SparseArrays.AbstractSparseMatrixCSC{T,<:Any}} where T

const AdjOrDense = Union{Matrix{T}, AdjDense{T}} where T
const AdjOrSparse = Union{<:SparseArrays.AbstractSparseMatrixCSC{T,<:Any}, AdjSparse{T}} where T



# TODO: The cost for materializing adjoints is very preliminary and should investigated in more detail
_copy_adjoint_cost(::Type{<:AdjOrDense}, A::MatrixInfo)::Float64 =
	prod(size(A))
_copy_adjoint_cost(::Type{<:AdjOrSparse}, A::MatrixInfo)::Float64 =
	(sum(size(A)) + prod(size(A))*A.pnz) * SPARSE_TIME_FACTOR
_copy_adjoint_cost(::Type{<:Diagonal}, A::MatrixInfo)::Float64 = 0
_copy_adjoint_cost(::Type{<:Any}, A::MatrixInfo)::Float64 =	Inf

_copy_adjoint(A::MatrixInfo) = _copy_adjoint_cost(A.type,A), MatrixInfo(reverse(A.sz),A.type,A.pnz)

function _copy_adjoint(S::SubResult)
	cost, matrixinfo = _copy_adjoint(S.matrixinfo)
	SubResult(cost+S.cost, S.pre_cost, matrixinfo, true, false, S.left, S.split, S.right, S.reverse_mul, S.child_adj)
end


function _make_dense(A::MatrixInfo)
	T = eltype(A.type)
	matrixtype = _is_adjoint(A) ? Adjoint{T,Matrix{T}} : Matrix{T}
	prod(A.sz), MatrixInfo(A.sz, matrixtype)
end

function _check_make_dense(S::SubResult)
	if S.matrixinfo.pnz < MAKE_DENSE_PNZ
		S
	else
		# NB: We make sure to *not* include the cost for _copy_adjoint here, since we get that for free when converting to dense
		copy_cost,matrixinfo = _make_dense(S.matrixinfo)
		SubResult(S.pre_cost+copy_cost, S.pre_cost, matrixinfo, S.copy_adj, true, S.left, S.split, S.right, S.reverse_mul, S.child_adj)
	end
end




_result_eltype(T1::DataType,T2::DataType) = promote_type(eltype(T1), eltype(T2))


function _indextype(T::DataType)
	T = _strip_adjoint(T)
	@assert T <: SparseArrays.AbstractSparseMatrixCSC
	T.parameters[2]
end
_result_indextype(T1::DataType,T2::DataType) = promote_type(_indextype(T1), _indextype(T2))



# Dense * Dense
function _chain(::Type{TA}, ::Type{TB}, A::MatrixInfo, B::MatrixInfo) where {TA<:AdjOrDense,TB<:AdjOrDense}
	cost = A.sz[1]*A.sz[2]*B.sz[2]
	# small adjustment to slightly prefer non-adjointed inputs (but cheaper than materializing the transpose)
	TA <: Adjoint && (cost += 0.25*prod(size(A)))
	TB <: Adjoint && (cost += 0.25*prod(size(B)))
	cost, MatrixInfo((A.sz[1],B.sz[2]), Matrix{_result_eltype(TA,TB)})
end

# Sparse * Dense
function _chain(::Type{TA}, ::Type{TB}, A::MatrixInfo, B::MatrixInfo) where {TA<:AdjOrSparse,TB<:AdjOrDense}
	P,N = size(A)
	M = size(B,2)
	cost = P*N*M*A.pnz * SPARSE_TIME_FACTOR
	# small adjustment to slightly prefer non-adjointed inputs (but cheaper than materializing the transpose)
	TB <: Adjoint && (cost += 0.25*prod(size(B)))
	cost, MatrixInfo((A.sz[1],B.sz[2]), Matrix{_result_eltype(TA,TB)})
end

# Dense * Sparse
function _chain(::Type{TA}, ::Type{TB}, A::MatrixInfo, B::MatrixInfo) where {TA<:AdjOrDense,TB<:AdjOrSparse}
	P,N = size(A)
	M = size(B,2)
	cost = P*N*M*B.pnz * SPARSE_TIME_FACTOR
	# small adjustment to slightly prefer non-adjointed inputs (but cheaper than materializing the transpose)
	TA <: Adjoint && (cost += 0.25*prod(size(A)))
	cost, MatrixInfo((A.sz[1],B.sz[2]), Matrix{_result_eltype(TA,TB)})
end

# Sparse * Sparse
function _chain(::Type{TA}, ::Type{TB}, A::MatrixInfo, B::MatrixInfo) where {TA<:AbstractSparseMatrixCSC,TB<:AbstractSparseMatrixCSC}
	# Rough estimate of the algorithmic complexity of the following algorithm:
	# For each column in B
	#   For each nonzero value in the column
	#   	Accumulate all values from one column in A
	#	Sort the values by row index
	#
	# A size P×N, B size N×M
	# a: probability of element being nonzero in A
	# b: probability of element being nonzero in B
	#
	# Collecting values for one output column: O(Nb+PNab))
	# Sorting one output column: O(PNab*log(PNab))
	# Doing both M times: O(PNMab + PNMab*log(PNab))

	P,N = size(A)
	M = size(B,2)

	cost = M*(N*B.pnz + P*N*A.pnz*B.pnz*(1+log2(max(2,P*N*A.pnz*B.pnz)))) * SPARSE_SPARSE_TIME_FACTOR
	pnz = 1 - (1-A.pnz*B.pnz)^N # Expected value of pnz given independent location of nonzeros

	matrixtype = SparseMatrixCSC{_result_eltype(TA,TB), _result_indextype(TA,TB)}
	cost, MatrixInfo((A.sz[1],B.sz[2]), matrixtype, pnz)
end


# Diagonal * X and X * Diagonal
# We do not include e.g. D*Bᵀ, since that materializes the adjoint
_chain(::Type{<:Diagonal}, ::Type{<:Matrix}, A::MatrixInfo, B::MatrixInfo) =
	prod(size(B)), B
_chain(::Type{<:Diagonal}, ::Type{<:AbstractSparseMatrixCSC}, A::MatrixInfo, B::MatrixInfo) =
	prod(size(B))*B.pnz*SPARSE_TIME_FACTOR, B
_chain(::Type{<:Matrix}, ::Type{<:Diagonal}, A::MatrixInfo, B::MatrixInfo) =
	prod(size(A)), A
_chain(::Type{<:AbstractSparseMatrixCSC}, ::Type{<:Diagonal}, A::MatrixInfo, B::MatrixInfo) =
	prod(size(A))*A.pnz*SPARSE_TIME_FACTOR, A
_chain(::Type{<:Diagonal}, ::Type{<:Diagonal}, A::MatrixInfo, B::MatrixInfo) =
	size(A,1), A



function _chain(::Type{<:Any}, ::Type{<:Any}, A::MatrixInfo, B::MatrixInfo)
	# throw(DomainError((A.type,B.type), "Unexpected matrix types"))
	Inf, MatrixInfo((A.sz[1],B.sz[2]), Nothing)
end


function _chain(A::MatrixInfo, B::MatrixInfo)
	@assert size(A,2)==size(B,1)
	_chain(A.type, B.type, A, B)
end



function _chain(A::SubResult, B::SubResult, reverse_mul, adjA, adjB, left, split, right)
	U = adjA ? _adjoint(A.matrixinfo) : A.matrixinfo
	V = adjB ? _adjoint(B.matrixinfo) : B.matrixinfo
	cost, matrixinfo = _chain(U,V)
	cost += A.cost + B.cost
	SubResult(cost, cost, matrixinfo, false, false, left, split, right, reverse_mul, (adjA,adjB))
end



function _chain(A::AdjointSparseChainResult, B::AdjointSparseChainResult, reverse_mul, adjA, adjB, left, split, right)
	a = getsubresult(A, adjA != reverse_mul)
	b = getsubresult(B, adjB != reverse_mul)
	_chain(a, b, reverse_mul, adjA, adjB, left, split, right)
end


function chain(A::AdjointSparseChainResult, B::AdjointSparseChainResult)
	# We have the following possibilities for computing the first SubResult
	# M = A.res*B.res
	# M = A.adj_res*B.res
	# M = A.res*B.adj_res'
	# M = A.adj_res'*B.adj_res'
	# M = copy(Mᵀ))             - if it's cheaper to compute Mᵀ and then adjoint!
	# And similarly for Mᵀ.

	left = A.res.left
	split = A.res.right
	right = B.res.right

	@assert left == A.adj_res.left
	@assert split == A.adj_res.right
	@assert right == B.adj_res.right

	@assert split+1==B.res.left
	@assert split+1==B.adj_res.left

	# @info "M"
	M1 = _chain(A, B, false, false, false, left, split, right) # AB
	M2 = _chain(A, B, false, true,  false, left, split, right) # Aᵀ'B
	M3 = _chain(A, B, false, false, true,  left, split, right) # ABᵀ'
	M4 = _chain(A, B, false, true,  true,  left, split, right) # Aᵀ'Bᵀ'

	M = M1
	M2.cost < M.cost && (M = M2)
	M3.cost < M.cost && (M = M3)
	M4.cost < M.cost && (M = M4)

	# @info "Mᵀ"
	MT1 = _chain(B, A, true, false, false, left, split, right) # BᵀAᵀ
	MT2 = _chain(B, A, true, true,  false, left, split, right) # B'Aᵀ
	MT3 = _chain(B, A, true, false, true,  left, split, right) # BᵀA'
	MT4 = _chain(B, A, true, true,  true,  left, split, right) # B'A'

	MT = MT1
	MT2.cost < MT.cost && (MT = MT2)
	MT3.cost < MT.cost && (MT = MT3)
	MT4.cost < MT.cost && (MT = MT4)

	adjM = _copy_adjoint(M)
	adjMT = _copy_adjoint(MT)

	M = _check_make_dense(M)
	MT = _check_make_dense(MT)
	adjM = _check_make_dense(adjM)
	adjMT = _check_make_dense(adjMT)

	if adjMT.cost < M.cost
		M = adjMT
	elseif adjM.cost < MT.cost
		MT = adjM
	end

	AdjointSparseChainResult(M,MT)
end


function merge_results!(results::Matrix{AdjointSparseChainResult}, i, j, R::AdjointSparseChainResult)
	A = min(results[i,j].res, R.res)
	B = min(results[i,j].adj_res, R.adj_res)

	results[i,j] = AdjointSparseChainResult(A,B)
	nothing
end


function plan_adjoint_sparse_chain(A::MatrixProduct, adj=false)
	results = optimize_chain(AdjointSparseChainResult, A.factors)
	order = chain_order(Union{AdjointSparseOperation,AdjointSparseCopyOperation}, results, A.factors, getsubresult(results[1,end],adj))
	results, order
end
function adjoint_sparse_chain_mul(A::MatrixProduct, adj=false)
	_,order = plan_adjoint_sparse_chain(A, adj)
	apply_chain(order)
end


# --- printing ---

function printnode_op(io::IO, op::AdjointSparseOperation)
	print(io, 'L')
	op.child_adj[1] && print(io, 'ᵀ')
	print(io, "*R")
	op.child_adj[2] && print(io, 'ᵀ')
end
function printnode_op(io::IO, op::AdjointSparseCopyOperation)
	print(io, op.make_dense ? "dense(A" : "copy(A")
	op.adj_in && print(io, 'ᵀ')
	print(io, ')')
	op.adj_out && print(io, 'ᵀ')
end

function print_op_compact(io::IO, co::ChainOrder, left, right, op::AdjointSparseOperation)
	print(io, '(')
	print_compact(io, (co,left))
	op.child_adj[1] && print(io, 'ᵀ')
	print_compact(io, (co,right))
	op.child_adj[2] && print(io, 'ᵀ')
	print(io, ')')
end

function print_op_compact(io::IO, co::ChainOrder, left, right, op::AdjointSparseCopyOperation)
	@assert right==0
	print(io, op.make_dense ? "dense(" : "copy(")
	print_compact(io, (co,left))
	op.adj_in && print(io, 'ᵀ')
	print(io, ')')
	op.adj_out && print(io, 'ᵀ')
end
