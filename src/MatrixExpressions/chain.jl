init_chain_element(::Type{T}, factors, i, j) where T =
	i==j ? T(factors[i],i) : T()

function init_chain(::Type{T}, factors) where T
	@assert all(x->x isa MatrixRef, factors) "Matrix chain optimization requires all factors to be of type MatrixRef."
	K = length(factors)
	[init_chain_element(T,factors,i,j) for i in 1:K, j in 1:K] # result matrix initialized with infinite costs
end


function optimize_chain(::Type{T}, factors) where T
	results = init_chain(T, factors)
	for j in 2:length(factors)
		for i in j-1:-1:1
			# we want to create the result for A.factors[i:j]
			for k in i:j-1
				res = chain(results[i,k], results[k+1,j])
				merge_results!(results, i, j, res)
			end
		end
	end
	results
end


struct ChainOrder{T1,T2}
	operations::Vector{Tuple{Int,Int,T1,Float64}} # For unary ops, the second index is zero
	cost::Float64
	factors::T2
end
ChainOrder(::Type{T1}, cost::Float64, factors::T2) where {T1,T2} = ChainOrder{T1,T2}(T1[],cost,factors)


getcost(R) = R.cost

function add_operations!(co, input1, input2, res)
	input2==0 && return input1
	push!(co.operations, (input1,input2,operation(res),getcost(res)))
	length(co.operations)
end

function _chain_order!(co::ChainOrder, results::Matrix, res)
	# depth first
	if isleaf(res)
		input1 = -getleafind(res)
		input2 = 0
	else # recurse
		input1 = _chain_order!(co, results, getleft(results,res))
		input2 = _chain_order!(co, results, getright(results,res))
	end
	add_operations!(co, input1, input2, res)
end


function chain_order(::Type{T}, results::Matrix, factors, res) where {T}
	# We will refer to the input matrices using negative indices in -K:-1
	# So that positive indices are used for multiplication results, and correspond to
	# indices in the order vector.
	co = ChainOrder(T, getcost(res), factors)
	_chain_order!(co, results, res)
	co
end


# depth first
function _apply_chain(co::ChainOrder, ind)
	l_ind,r_ind,op,_ = co.operations[ind]

	L = l_ind < 0 ? co.factors[-l_ind].matrix : _apply_chain(co, l_ind)

	X = if r_ind==0
		op(L)
	else
		R = r_ind < 0 ? co.factors[-r_ind].matrix : _apply_chain(co, r_ind)
		op(L,R)
	end
	# @info "applied op $(typeof(op))"
	# @show typeof(X)
	# @show size(X)
	X
end

function apply_chain(co::ChainOrder, ind=length(co.operations))
	if ind==0
		only(co.factors).matrix
	else
		_apply_chain(co, ind)
	end
end



# --- printing ---


function AbstractTrees.children((co,ind)::Tuple{ChainOrder,Int})
	ind<0 && return ()
	left,right,_ = co.operations[ind]
	right != 0 ? ((co,left),(co,right)) : ((co,left),)
end


# Fallback definition
printnode_op(io::IO, op) = print(io, op)

function AbstractTrees.printnode(io::IO, (co,ind)::Tuple{ChainOrder,Int})
	if ind<0
		print(co.factors[-ind])
	else
		left,right,op,cost = co.operations[ind]
		printnode_op(io, op)
		print(io, " (cost=", round(cost,digits=2), ')')
	end
end


# Fallback definition
function print_op_compact(io::IO, co::ChainOrder, left, right, op)
	if right==0
		print(io, nameof(typeof(op)), '(')
		print_compact(io, (co,left))
		print(io, ')')
	else
		print(io, '(')
		print_compact(io, (co,left))
		print(io, '[', nameof(typeof(op)), ']')
		print_compact(io, (co,right))
		print(io, ')')
	end
end

function print_compact(io::IO, (co,ind)::Tuple{ChainOrder,Int})
	if ind<0
		print(io, co.factors[-ind])
	else
		left,right,op,_ = co.operations[ind]
		print_op_compact(io, co, left, right, op)
	end
end


function Base.show(io::IO, co::ChainOrder)
	L = length(co.operations)
	tree = (co,L)
	if get(io,:compact,false)
		L>0 && print_compact(io, tree)
	else
		print(io, "Total cost: ", co.cost)
		if L>0
			println(io)
			AbstractTrees.print_tree(io, tree)
		end
	end
end
