using SingleCellProjections
using .SingleCellProjections.MatrixExpressions
using LinearAlgebra
using SparseArrays
using Random

using .MatrixExpressions: ChainOrder,
                          adjoint_sparse_chain_mul, plan_adjoint_sparse_chain,
                          plan_diag_chain,
                          apply_chain,
                          MatrixInfo,
                          _is_dense,
                          MAKE_DENSE_PNZ


include("reference.jl")


mr(;kwargs...) = MatrixRef(kwargs...)
ms(args...)    = matrixsum(args...)
ms(;kwargs...) = matrixsum(kwargs...)
mp(args...)    = matrixproduct(args...)
mp(;kwargs...) = matrixproduct(kwargs...)
di(;kwargs...) = Diag(mp(;kwargs...))
dg(;kwargs...) = DiagGram(mp(;kwargs...))

adjconvert(::Type{T}, X) where T = convert(T,X)
adjconvert(::Type{<:Adjoint{<:Any,T}}, X) where T = convert(T,X')'


shortname(::Type) = "Unknown"
shortname(::Type{<:Adjoint}) = "Adjoint"
shortname(::Type{<:Matrix}) = "Matrix"
shortname(::Type{<:Adjoint{<:SparseArrays.AbstractSparseMatrixCSC}}) = "AdjSparse"
shortname(::Type{<:SparseArrays.AbstractSparseMatrixCSC}) = "Sparse"
shortname(args::Type...) = join(shortname.(args),'_')
shortname(args...) = shortname(typeof.(args)...)

function orderstring(order::ChainOrder)
    io = IOBuffer()
    print(IOContext(io, :compact=>true), order)
    String(take!(io))
end
orderstring(A::MatrixProduct, adj=false) = orderstring(plan_adjoint_sparse_chain(A,adj)[2])
orderstring(D::Diag) = orderstring(plan_diag_chain(D.A)[2])


# sprand but with exactly specified number of nonzero elements
function sprand_nnz(rng,h,w,nnz)
    @assert 0<=nnz<=h*w
    ind = randperm(rng, h*w)[1:nnz]
    I = mod1.(ind,h)
    J = fld1.(ind,h)
    V = rand(rng,nnz)
    sparse(I,J,V,h,w)
end



@testset "MatrixExpressions.jl" begin
    @testset "Basic" begin
        A = [1 6 2; 4 5 3; 9 2 4; 6 1 5]
        B = [5 0; 2 2; 9 2]
        C = [7 2 0 5 1; 8 8 0 2 3]
        D = reshape(collect(1:20),5,4)
        E = [2 0 4 1 7; 2 3 4 0 1; 6 7 3 2 2]
        F = [7 4 1 9; 6 6 4 4; 4 1 6 2; 5 9 6 3; 6 4 5 2]
        AB = A*B
        ABC = AB*C
        ABCD = AB*C*D

        @testset "Mul" begin
            @test compute(mp(;A)) == A
            let R = adjoint_sparse_chain_mul(mp(;A),true)
                @test R == A'
                @test R isa Matrix
            end

            @testset "$(shortname(a,b))" for a in (A,Matrix(A')'), b in (B,Matrix(B')')
                @test compute(mp(;a,b)) == AB
            end

            @testset "$(shortname(a,b,c))" for a in (A,Matrix(A')'), b in (B,Matrix(B')'), c in (C,Matrix(C')')
                @test compute(mp(;a,b,c)) == ABC
            end
        end

        @testset "MulSumMul" begin
            @test compute(matrixproduct(:A=>A, matrixsum(mp(;B,C),:E=>E), :D=>D)) == A*(B*C+E)*D
        end

        @testset "Diag" begin
            @test compute(di(;A,B,C,D)) == diag(ABCD)

            @test compute(Diag(matrixproduct(:A=>A, matrixsum(mp(;B,C),:E=>E), :D=>D))) == diag(A*(B*C+E)*D)
            @test compute(Diag(matrixproduct(:A=>A, matrixsum(mp(;B,C),:E=>E), matrixsum(:D=>D,:F=>F)))) == diag(A*(B*C+E)*(D+F))
        end

        @testset "DiagGram" begin
            @test compute(dg(;A)) == diag(A'A)
            @test compute(dg(;A=A')) == diag(A*A')

            @testset "$(shortname(a,b))" for a in (A,Matrix(A')'), b in (B,Matrix(B')')
                @test compute(dg(;a,b)) == diag(AB'AB)
            end
        end

        @testset "Order" begin
            @test orderstring(mp(;A,B)) == "(AB)"
            @test orderstring(mp(;A,B,C)) == "((AB)C)"
            @test orderstring(mp(;C=C',B=B',A=A')) == "(C'(A'ᵀB'ᵀ)ᵀ)"

            @test orderstring(di(;A,B,C,D)) == "diag((BᵀAᵀ)ᵀ(CD))"
            @test orderstring(di(;A=Matrix(A')',B=Matrix(B')',C,D)) == "diag((B'ᵀA'ᵀ)ᵀ(CD))"
        end
    end

    @testset "copy" begin
        X = mr(;X=zeros(2,2))
        Y = mr(;Y=ones(2,2))
        Z = mr(;Z=[1 2; 3 4]')

        @testset "MatrixRef" begin
            X2 = copy(X)
            @test X2 === X
        end

        @testset "MatrixSum" begin
            s = ms(X,Y)
            s2 = copy(s)
            @test s2.terms !== s.terms
            @test s2.terms == s.terms
        end

        @testset "MatrixProduct" begin
            p = mp(X,Y)
            p2 = copy(p)
            @test p2.factors !== p.factors
            @test p2.factors == p.factors
        end

        @testset "Recursive" begin
            p1 = mp(X,Y)
            p2 = mp(Z,Y)
            s = ms(p1,p2)
            o = mp(Y,s,X)

            o2 = copy(o)
            @test o2.factors !== o.factors
            @test o2.factors == o.factors

            @test o2.factors[1] === Y
            @test o2.factors[3] === X

            @test o2.factors[2] !== s
            @test o2.factors[2] == s

            @test o2.factors[2].terms[1] !== p1
            @test o2.factors[2].terms[1] == p1
            @test o2.factors[2].terms[2] !== p2
            @test o2.factors[2].terms[2] == p2
        end

        @testset "Diag" begin
            p = mp(X,Y)
            d = Diag(p)
            d2 = copy(d)

            @test d.A !== d2.A
            @test d.A == d2.A
        end

        @testset "DiagGram" begin
            p = mp(X,Y)
            dg = DiagGram(p)
            dg2 = copy(dg)

            @test dg.A !== dg2.A
            @test dg.A == dg2.A
        end
    end

    @testset "Reference" begin
        # Quite carefully chosen to test different scenarios for multiplication order and converting to dense
        nnzs = ((3,8,5), (5,5,12), (5,15,12), (7,17,13), (5,15,8), (2,2,20))

        @testset "nnz=($nnzA,$nnzB,$nnzC)" for (nnzA,nnzB,nnzC) in nnzs
            rng = StableRNG(9512)

            A = sprand_nnz(rng, 3, 5, nnzA)
            B = sprand_nnz(rng, 5, 8, nnzB)
            C = sprand_nnz(rng, 8, 4, nnzC)
            pnzA = nnzA/length(A)
            pnzB = nnzB/length(B)
            pnzC = nnzC/length(C)

            AB = A*B
            AB_adj = copy(AB')
            ABC = AB*C
            ABC_adj = copy(ABC')

            types = (SparseMatrixCSC, Adjoint{<:Any,SparseMatrixCSC}, Matrix, Adjoint{<:Any,Matrix})

            @testset "$(shortname(TA,TB))" for TA in types, TB in types
                _is_dense(TA) && pnzA<MAKE_DENSE_PNZ && continue # To avoid duplicate tests of the same thing
                _is_dense(TB) && pnzB<MAKE_DENSE_PNZ && continue # To avoid duplicate tests of the same thing

                a = adjconvert(TA,A)
                b = adjconvert(TB,B)

                p = mp(;a,b)
                ref = ref_optimize(p, false)
                _, order = plan_adjoint_sparse_chain(p, false)
                @test 0 < order.cost < Inf
                @test order.cost ≈ ref.cost
                @test orderstring(order) == ref.expression
                @test apply_chain(order) ≈ AB

                ref_adj = ref_optimize(p, true)
                _, order_adj = plan_adjoint_sparse_chain(p, true)
                @test 0 < order_adj.cost < Inf
                @test order_adj.cost ≈ ref_adj.cost
                @test orderstring(order_adj) == ref_adj.expression
                @test apply_chain(order_adj) ≈ AB_adj
            end

            @testset "$(shortname(TA,TB,TC))" for TA in types, TB in types, TC in types
                _is_dense(TA) && pnzA<MAKE_DENSE_PNZ && continue # To avoid duplicate tests of the same thing
                _is_dense(TB) && pnzB<MAKE_DENSE_PNZ && continue # To avoid duplicate tests of the same thing
                _is_dense(TC) && pnzC<MAKE_DENSE_PNZ && continue # To avoid duplicate tests of the same thing

                a = adjconvert(TA,A)
                b = adjconvert(TB,B)
                c = adjconvert(TC,C)

                p = mp(;a,b,c)
                ref = ref_optimize(p, false)
                _, order = plan_adjoint_sparse_chain(p, false)
                @test 0 < order.cost < Inf
                @test order.cost ≈ ref.cost
                @test orderstring(order) == ref.expression
                @test apply_chain(order) ≈ ABC

                ref_adj = ref_optimize(p, true)
                _, order_adj = plan_adjoint_sparse_chain(p, true)
                @test 0 < order_adj.cost < Inf
                @test order_adj.cost ≈ ref_adj.cost
                @test orderstring(order_adj) == ref_adj.expression
                @test apply_chain(order_adj) ≈ ABC_adj
            end
        end
    end
end
