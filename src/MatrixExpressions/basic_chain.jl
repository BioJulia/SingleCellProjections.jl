struct BasicChainResult
	cost::Float64
	sz::Tuple{Int,Int}
	# How it was constructed
	left::Int
	split::Int # 0 used for leaves
	right::Int
end
BasicChainResult() = BasicChainResult(Inf, (0,0), 0, 0, 0)
BasicChainResult(A::AbstractMatrix, leaf_ind::Int) = BasicChainResult(0.0, size(A), leaf_ind, 0, leaf_ind)
BasicChainResult(A::MatrixRef,leaf_ind::Int) = BasicChainResult(A.matrix, leaf_ind)

Base.size(R::BasicChainResult) = R.sz
Base.size(R::BasicChainResult, ind) = ind>2 ? 1 : R.sz[ind]

struct BasicChainOperation end
operation(::BasicChainResult) = BasicChainOperation()
(::BasicChainOperation)(A,B) = A*B


isleaf(R::BasicChainResult) = R.split==0
getleafind(R::BasicChainResult) = R.left
getleft(results, R::BasicChainResult) = results[R.left,R.split]
getright(results, R::BasicChainResult) = results[R.split+1,R.right]


# function chain(A::BasicChainResult, B::BasicChainResult, left, split, right)
function chain(A::BasicChainResult, B::BasicChainResult)
	@assert size(A,2)==size(B,1)
	@assert A.right+1==B.left
	cost = A.cost + B.cost + size(A,1)*size(A,2)*size(B,2)
	sz = (size(A,1),size(B,2))
	BasicChainResult(cost, sz, A.left, A.right, B.right)
end


function merge_results!(results::Matrix{BasicChainResult}, i, j, res::BasicChainResult)
	if res.cost < results[i,j].cost
		results[i,j] = res
	end
	nothing
end


function plan_basic_chain(A::MatrixProduct)
	results = optimize_chain(BasicChainResult,A.factors)
	order = chain_order(BasicChainOperation, results, A.factors, results[1,end])
	results, order
end
function basic_chain_mul(A::MatrixProduct)
	_,order = plan_basic_chain(A)
	apply_chain(order)
end


# --- printing ---

printnode_op(io::IO, op::BasicChainOperation) = print(io, '*')
function print_op_compact(io::IO, co::ChainOrder, left, right, op::BasicChainOperation)
	print(io, '(')
	print_compact(io, (co,left))
	print_compact(io, (co,right))
	print(io, ')')
end
