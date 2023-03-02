isgz(fn) = lowercase(splitext(fn)[2])==".gz"
_open(f, fn) = open(fn) do io
    f(isgz(fn) ? GzipDecompressorStream(io) : io)
end

read_matrix(fn,delim=',') = _open(io->readdlm(io,delim,Int), fn)
read_strings(fn,delim=',') = _open(io->readdlm(io,delim,String), fn)

function simple_logtransform(X, scale_factor)
	s = sum(X; dims=1)
	log2.( 1 .+ X.*scale_factor./max.(1,s) )
end

function simple_idf(X)
	df = vec(max.(mean(X; dims=2), 1/size(X,2)))
	1 ./ df
end

function simple_tf_idf_transform(X, idf, scale_factor)
	@assert size(idf) == (size(X,1),)
	# log( 1 + c*f/s * idf )
	s = max.(1,sum(X;dims=1))
	log.(1 .+ X.*scale_factor.*idf./s )
end


materialize(X) = X
materialize(X::MatrixExpression) = X*I(size(X,2))
materialize(X::SVD) = convert(Matrix,X)
materialize(X::SingleCellProjections.LowRank) = X.U*X.Vt
materialize(X::DataMatrix) = materialize(X.matrix)

function pairwise_dist(X)
	K = X'X
	d = diag(K)
	D2 = d .+ d' .- 2.0.*K
	sqrt.(max.(0.0, D2))
end

function ncommon_neighbors(A,B; k=20)
	@assert size(A,2)==size(B,2)
	N = size(A,2)
	Dr = pairwise_dist(A)
	Df = pairwise_dist(B)
	ncommon = zeros(Int,N)
	for i in 1:N
		a = sortperm(Dr[:,i])[1:k]
		b = sortperm(Df[:,i])[1:k]
		ncommon[i] = length(intersect(a,b))
	end
	ncommon
end

function test_show(data::DataMatrix; matrix=nothing, var=nothing, obs=nothing, models=nothing)
    io = IOBuffer()
    show(io, MIME("text/plain"), data)
    str = String(take!(io))
    s = split(str, '\n')
    @test 4<=length(s)<=5
    @test s[1] == "DataMatrix ($(size(data,1)) variables and $(size(data,2)) observations)"
    matrix!==nothing && @test contains(s[2][3:end], matrix)
    @test startswith(s[3],"  Variables: ")
    var!==nothing && @test sort(split(s[3][14:end], ", "))==sort(var)
    @test startswith(s[4],"  Observations: ")
    obs!==nothing && @test sort(split(s[4][17:end], ", "))==sort(obs)

    length(s)>4 && @test startswith(s[5],"  Models: ")
    if models !== nothing
        m = replace(get(s,5,""),"  Models: "=>"")
        @test contains(m, models)
    end
end
