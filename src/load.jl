function samplenamesfromfilenames(filenames)::Vector{String}
	n = first.(splitext.(filenames))
	n = replace.(n, r"_?(matrix|filtered_feature_bc_matrix|sample_feature_bc_matrix|sample_filtered_feature_bc_matrix)$"i=>"")

	s = splitpath.(n)
	length(filenames)==1 && return [last(s[1])]
	for i=1:minimum(length.(s))
		curr = last.(s)
		if !all(isequal(curr[1]), curr)
			@assert allunique(curr) "Failed to find unique sample names from filenames: $filenames"
			return curr
		end
	end
	error("Failed to find unique sample names from filenames: $filenames")
end


struct Lazy10xMatrix{Tv,Ti}
	sz::Tuple{Int,Int}
	nnz::Int
	filename::String
end
Lazy10xMatrix(::Type{Tv},::Type{Ti}, args...) where {Tv,Ti} = Lazy10xMatrix{Tv,Ti}(args...)

struct LazyMergedMatrix{Tv,Ti}
	sz::Tuple{Int,Int}
	nnz::Int
	data::Vector{DataMatrix}
	var_id_cols
end
LazyMergedMatrix(::Type{Tv},::Type{Ti}, args...) where {Tv,Ti} = LazyMergedMatrix{Tv,Ti}(args...)


Base.size(matrix::Lazy10xMatrix) = matrix.sz
Base.size(matrix::Lazy10xMatrix, dim) = dim>2 ? 1 : matrix.sz[dim]
SparseArrays.nnz(matrix::Lazy10xMatrix) = matrix.nnz

Base.size(matrix::LazyMergedMatrix) = matrix.sz
Base.size(matrix::LazyMergedMatrix, dim) = dim>2 ? 1 : matrix.sz[dim]
SparseArrays.nnz(matrix::LazyMergedMatrix) = matrix.nnz


Base.eltype(matrix::Lazy10xMatrix{Tv}) where Tv = Tv
_indextype(::Lazy10xMatrix{<:Any,Ti}) where {Ti} = Ti
_indextype(::AbstractSparseMatrix{<:Any,Ti}) where {Ti} = Ti



getmatrix(matrix) = matrix
getmatrix(matrix::Lazy10xMatrix{Tv,Ti}) where {Tv,Ti} = read10x_matrix(matrix.filename, SparseMatrixCSC{Tv,Ti})


load_counts(data::DataMatrix; callback=nothing) = data
load_counts(data::DataMatrix{Lazy10xMatrix}; callback=nothing) = DataMatrix(getmatrix(data.matrix), data.var, data.obs)
function load_counts(data::DataMatrix{LazyMergedMatrix{Tv,Ti}}; callback=nothing) where {Tv,Ti}
	lazy_matrix = data.matrix

	sample_features = getfield.(lazy_matrix.data,:var)
	matrices = getfield.(lazy_matrix.data,:matrix)
	matrix = _merge_matrices(Tv, Ti, data.var, sample_features, matrices; lazy_matrix.var_id_cols, callback)
	matrix===nothing && return nothing
	update_matrix(data, matrix; var=:keep, obs=:keep)
end



function _load10x_metadata(io)
	features = read10x_features(io, DataFrame)
	cells = read10x_barcodes(io, DataFrame)
	P,N,nz = read10x_matrix_metadata(io)
	P,N,nz,features,cells
end

function load10x(filename; lazy=false, copy_obs_cols="barcode"=>"id", kwargs...)
	if lazy
		if lowercase(splitext(filename)[2]) == ".h5"
			P,N,nz,features,cells = h5open(_load10x_metadata, filename)
		else
			P,N,nz,features,cells = _load10x_metadata(filename)
		end
		matrix = Lazy10xMatrix(Int, Int32, (P,N), nz, filename)
	else
		matrix, features, cells = read10x(filename, SparseMatrixCSC{Int,Int32}, DataFrame, DataFrame)
	end

	if copy_obs_cols !== nothing
		cells = hcat(select(cells, copy_obs_cols), cells)
	end

	DataMatrix(matrix, features, cells; kwargs...)
end

function load_counts(loadfun, filenames;
                     lazy=loadfun==load10x,
                     lazy_merge=false,
                     var_id_cols=nothing,#["id","feature_type"], # nothing means merge from samples
                     sample_names=samplenamesfromfilenames(filenames),
                     sample_name_col = sample_names===nothing ? nothing : "sampleName",
                     merged_obs_id_col = "id",
                     merged_obs_id_delim = '_',
                     callback=nothing)

	# TODO: call callback between sample loads(?)
	args1 = lazy ? (;lazy) : (;)
	args2 = var_id_cols !== nothing ? (;var_id_cols) : (;)
	samples = loadfun.(filenames; args1..., args2...) # Do *not* pass kwargs to loadfuns that might not support them

	merge_counts(samples, sample_names; lazy=lazy_merge, var_id_cols, sample_name_col, merged_obs_id_col, merged_obs_id_delim, callback)
end

# default to 10x
load_counts(filenames; kwargs...) = load_counts(load10x, filenames; kwargs...)



_value_or_ambiguous(x) = Ref(length(unique(x))!=1 ? "ambiguous" : first(x))


function _merge_features(features; var_id_cols)
	c = coalesce.(vcat(features..., cols=:union),"")
	g = groupby(c, var_id_cols)

	# Bug in DataFrames? if valuecols(g) is empty, an empty DataFrame is returned, but we want the keys.
	# combine(g, valuecols(g) .=> _value_or_ambiguous; renamecols=false)

	# Workaround.
	cols = vcat(Symbol.(var_id_cols), valuecols(g))
	combine(g, cols .=> _value_or_ambiguous; renamecols=false)
end

function _merge_cells(samples, sample_names; sample_name_col, merged_obs_id_col, merged_obs_id_delim)
	x = unique(getfield.(samples,:obs_id_cols))
	length(x) != 1 && error("Cannot merge samples with different \"obs_id_cols\".")
	obs_id_cols = only(x)


	sample_cells = DataFrame[]
	for (k,s) in enumerate(samples)
		c = copy(s.obs; copycols=false) # share vectors, but make it possible to add columns

		sample_name_col !== nothing && insertcols!(c, 1, sample_name_col=>sample_names[k])

		if merged_obs_id_col !== nothing
			prefix = sample_names != nothing ? string(sample_names[k], merged_obs_id_delim) : ""
			_join(args...) = string(prefix, join(args, merged_obs_id_delim))
			transform!(c, obs_id_cols=>((args...)->_join.(args...))=>merged_obs_id_col)
			select!(c, merged_obs_id_col, Not(merged_obs_id_col)) # ensure :id is the first column
		end

		push!(sample_cells, c)
	end

	if merged_obs_id_col !== nothing
		obs_id_cols = [merged_obs_id_col]
	elseif sample_name_col !== nothing
		obs_id_cols = vcat(sample_name_col, obs_id_cols)
	end

	cells = vcat(sample_cells...; cols=:union)

	size(unique(select(cells,obs_id_cols)),1) != size(cells,1) && error("Merged cell ids are not unique.")

	cells, obs_id_cols
end


function _insert_matrix!(colptr, rowval, nzval, colptr_offset, nnz_offset, feature_ind, sf, var_id_cols, A)
	cp = SparseArrays.getcolptr(A)
	rv = rowvals(A)
	nz = nonzeros(A)

	colptr[colptr_offset .+ (1:size(A,2)+1)] .= cp .+ nnz_offset

	# Figure out how to map feature indices in sample to global feature indices
	sf_ind = select(sf, var_id_cols)
	leftjoin!(sf_ind, feature_ind; on=var_id_cols)
	row_ind = sf_ind.__row__

	rowval[nnz_offset .+ (1:nnz(A))] .= getindex.(Ref(row_ind),rv)
	nzval[nnz_offset .+ (1:nnz(A))] .= nz
end



function _merge_matrices(::Type{Tv}, ::Type{Ti}, features, sample_features, matrices;
                         var_id_cols, callback) where {Tv,Ti}
	P = size(features,1)
	N = sum(A->size(A,2), matrices)
	nnonzeros = sum(nnz, matrices)


	feature_ind = select(features, var_id_cols)
	feature_ind.__row__ = 1:P

	colptr = Vector{Ti}(undef, N+1)
	rowval = Vector{Ti}(undef, nnonzeros)
	nzval  = Vector{Tv}(undef, nnonzeros)

	colptr_offset = 0
	nnz_offset = 0
	for (sf,matrix) in zip(sample_features, matrices)
		callback !== nothing && callback() && return nothing

		A = getmatrix(matrix)::AbstractSparseMatrix # handle lazy loading
		_insert_matrix!(colptr, rowval, nzval, colptr_offset, nnz_offset, feature_ind, sf, var_id_cols, A) # function barrier to handle type instabilities

		colptr_offset += size(A,2)
		nnz_offset += nnz(A)
	end

	SparseMatrixCSC(P, N, colptr, rowval, nzval)
end


# If sample_names are provided, they *will* be used to create the new IDs, either merging or adding to obs_id_cols.
# merged_obs_id_col = nothing, the current obs_id_cols will be reused (+ sampleNames if provided).
function merge_counts(samples, sample_names;
                      lazy=false,
                      var_id_cols=nothing, #["id","feature_type"], # nothing means autodetect
                      sample_name_col = sample_names===nothing ? nothing : "sampleName",
                      merged_obs_id_col = "id",
                      merged_obs_id_delim = '_',
                      callback=nothing)
	@assert sample_name_col===nothing || length(samples)==length(sample_names)

	sample_features = getfield.(samples,:var)

	if var_id_cols === nothing
		c = getfield.(samples,:var_id_cols)
		# merge the var_id_cols from each sample
		var_id_cols = union(Iterators.flatten(c))

		@assert all(isequal(first(var_id_cols)), first.(c)) "Variable IDs do not match between sample files."
		@assert length(var_id_cols)<=2
		if length(var_id_cols)>1
			@assert all(x->length(x)==1 || x[2]==var_id_cols[2], c)
		end
	end

	features = _merge_features(sample_features; var_id_cols)
	cells, obs_id_cols = _merge_cells(samples, sample_names; sample_name_col, merged_obs_id_col, merged_obs_id_delim)

	matrices = getfield.(samples,:matrix)
	Tv = eltype(first(matrices))
	Ti = _indextype(first(matrices))

	P = size(features,1)
	N = sum(A->size(A,2), matrices)
	nnonzeros = sum(nnz, matrices)

	lazy_matrix = LazyMergedMatrix(Tv, Ti, (P,N), nnonzeros, convert(Vector{DataMatrix},samples), var_id_cols)
	lazy_data = DataMatrix(lazy_matrix, features, cells; var_id_cols, obs_id_cols)
	lazy ? lazy_data : load_counts(lazy_data; callback)
end
