import .MatrixExpressions: MatrixInfo, _chain, _is_adjoint, _adjoint, _copy_adjoint, _make_dense, MAKE_DENSE_PNZ

struct Factor
	cost::Float64
	matrixinfo::MatrixInfo
	expression::String
end
Factor() = Factor(Inf,MatrixInfo(),"")
Factor(X::MatrixRef) = Factor(0.0, MatrixInfo(X.matrix), string(X.name, X.matrix isa Adjoint ? "'" : ""))


function mul(A::Factor, B::Factor)
	c,m = _chain(A.matrixinfo, B.matrixinfo)
	c += A.cost + B.cost
	expression = string('(',A.expression,B.expression,')')
	Factor(c,m,expression)
end

_adjoint(A::Factor) = Factor(A.cost, _adjoint(A.matrixinfo), string(A.expression,'ᵀ'))

function _copy_adjoint(A::Factor)
	c,m = _copy_adjoint(A.matrixinfo)
	adj = _is_adjoint(A.matrixinfo)
	Factor(A.cost+c, m, string("copy(", A.expression, (adj ? "" : "ᵀ"), ')', (adj ? "ᵀ" : "")))
end

function _check_make_dense(A::Factor, base_cost::Float64=A.cost)
	if A.matrixinfo.pnz<MAKE_DENSE_PNZ
		A
	else
		c,m = _make_dense(A.matrixinfo)
		if startswith(A.expression, "copy(")
			expression = string("dense", A.expression[5:end]) # replace "copy" with "dense"
		else
			if _is_adjoint(A.matrixinfo)
				expression = string("dense(", A.expression, "ᵀ)ᵀ")
			else
				expression = string("dense(", A.expression, ')')
			end
		end
		Factor(base_cost+c, m, expression)
	end
end




function ref_optimize_mul!(d, A, B, adjA, adjB)
	fA = ref_optimize!(d, A, adjA)
	fB = ref_optimize!(d, B, adjB)
	fA = adjA ? _adjoint(fA) : fA
	fB = adjB ? _adjoint(fB) : fB
	mul(fA,fB)
end

function ref_optimize_reversemul!(d, A, B, adjA, adjB)
	fA = ref_optimize!(d, A, adjA)
	fB = ref_optimize!(d, B, adjB)
	fA = adjA ? fA : _adjoint(fA)
	fB = adjB ? fB : _adjoint(fB)
	mul(fB,fA)
end


function ref_optimize_single!(d, factors, adj)
	get!(d, (factors,adj)) do
		if length(factors)==1
			F = only(factors)
			# make_dense = F.matrixinfo.pnz >= MAKE_DENSE_PNZ
			# make_dense && (F = _make_dense(F))
			# adj && (F = _copy_adjoint(F, make_dense))
			base_cost = F.cost # Maybe just use 0.0 which it should be?
			adj && (F = _copy_adjoint(F))
			# F.matrixinfo.pnz >= MAKE_DENSE_PNZ && (F = _make_dense(F))
			F = _check_make_dense(F,base_cost)
			F
		else
			F_opt = Factor()

			for i in 1:length(factors)-1
				A = @view factors[1:i]
				B = @view factors[i+1:end]

				if adj
					F = ref_optimize_reversemul!(d, A, B, false, false)
					F.cost < F_opt.cost && (F_opt = F)

					F = ref_optimize_reversemul!(d, A, B, true, false)
					F.cost < F_opt.cost && (F_opt = F)

					F = ref_optimize_reversemul!(d, A, B, false, true)
					F.cost < F_opt.cost && (F_opt = F)

					F = ref_optimize_reversemul!(d, A, B, true, true)
					F.cost < F_opt.cost && (F_opt = F)
				else
					F = ref_optimize_mul!(d, A, B, false, false)
					F.cost < F_opt.cost && (F_opt = F)

					F = ref_optimize_mul!(d, A, B, true, false)
					F.cost < F_opt.cost && (F_opt = F)

					F = ref_optimize_mul!(d, A, B, false, true)
					F.cost < F_opt.cost && (F_opt = F)

					F = ref_optimize_mul!(d, A, B, true, true)
					F.cost < F_opt.cost && (F_opt = F)
				end
			end

			@assert F_opt.cost < Inf

			F_opt
		end
	end
end


# responsible for making results dense and copy adjoints
function ref_optimize!(d, factors, adj)
	haskey(d, (factors,adj)) && return d[(factors,adj)]

	F1 = ref_optimize_single!(d, factors, adj)
	F2 = ref_optimize_single!(d, factors, !adj)

	F1T = _copy_adjoint(F1)
	F2T = _copy_adjoint(F2)

	F1T = _check_make_dense(F1T, F1.cost)
	F2T = _check_make_dense(F2T, F2.cost)
	F1 = _check_make_dense(F1)
	F2 = _check_make_dense(F2)

	if F2T.cost < F1.cost
		F1 = F2T
	elseif F1T.cost < F2.cost
		F2 = F1T
	end

	d[(factors,adj)] = F1
	d[(factors,!adj)] = F2

	F1
end

ref_optimize(p::MatrixProduct, adj) = ref_optimize!(Dict(),Factor.(p.factors),adj)
