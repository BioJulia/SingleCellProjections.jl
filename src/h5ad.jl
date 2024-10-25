function _read_h5_attribute(g, name, default)
	a = attributes(g)
	haskey(a, name) ? read(a,name) : default
end
_read_h5_attribute(g, name) = read(attributes(g), name)


_readh5ad_dataframe_string_array(g) = read(g)
_readh5ad_dataframe_array(g) = read(g)

function _readh5ad_dataframe_categorical(g)
	ordered = read(attributes(g),"ordered")
	@assert ordered == false "TODO: support ordered=true"

	categories = read(g["categories"])
	codes = read(g["codes"])
	codes .+= 1

	categories[codes] # TODO: use some kind of pooled representation?
end


function _readh5ad_dataframe_column(g; anndata_version)
	if anndata_version === v"0.0.0"
		# TODO: handle categorical data etc...
		read(g)
	else
		et = read(attributes(g),"encoding-type")
		if et == "string-array"
			_readh5ad_dataframe_string_array(g)
		elseif et == "array"
			_readh5ad_dataframe_string_array(g)
		elseif et == "categorical"
			_readh5ad_dataframe_categorical(g)
		else
			error("Unknown column type: ", et)
		end
	end
end


function _readh5ad_dataframe(g; id_column, anndata_version)
	t = _read_h5_attribute(g, "encoding-type", "dataframe")
	@assert t == "dataframe" "Expected $g to be a dataframe, got \"$t\""
	v = _read_h5_attribute(g, "encoding-version", "0.1.0")
	@assert v in ("0.1.0","0.2.0") "Expected $g to have version 0.1.0 or 0.2.0, got \"$v\""

	_index = _read_h5_attribute(g,"_index")
	id = _readh5ad_dataframe_column(g[_index]; anndata_version)

	df = DataFrame(id_column=>id)

	
	column_order = _read_h5_attribute(g,"column-order")
	if !isempty(column_order)
		columns = [_readh5ad_dataframe_column(g[c]; anndata_version) for c in column_order]

		df2 = DataFrame(column_order .=> columns)
		df = hcat(df,df2)
	end
	df
end



function _try_convert_array(T, data)
	eltype(data) <: T && return data
	try
		return T.(data)
	catch ex
		ex isa InexactError && return nothing
		rethrow(ex)
	end
end


function _fix_sparse_buffers!(P,N,indptr,rowval,nzval)
	@assert indptr[1] == 1
	@assert indptr[end] == length(rowval)+1

	for j in 1:N
		rng = indptr[j]:indptr[j+1]-1
		isempty(rng) && continue

		rowval_j = @view rowval[rng]
		if !issorted(rowval_j) # These are normally sorted - but I've found cellranger(?) .h5 files in the wild where they are not. So better check or the data will be corrupt.
			nzval_j = @view nzval[rng]

			perm = sortperm(rowval_j)
			rowval_j .= rowval_j[perm]
			nzval_j .= nzval_j[perm]
		end
		@assert rowval_j[1] >= 1
		@assert rowval_j[end] <= P
	end
end


function _readh5ad_sparse(::Type{T}, g, trans=false) where T
	_read_h5_attribute(g, "encoding-version") == "0.1.0" || @warn "Expected sparse encoding-version to be \"0.1.0\""
	shape = _read_h5_attribute(g, "shape")
	@assert length(shape) == 2

	N,P = shape

	data = read(g, "data")
	data2 = _try_convert_array(T, data)
	if data2 === nothing
		@warn "Failed to convert matrix \"$(HDF5.name(g))\" eltype from $(eltype(data)) to $T, skipping."
		return nothing
	end
	data = data2

	indices = convert(Vector{Int32}, read(g,"indices"))
	indptr = convert(Vector{Int32}, read(g,"indptr"))

	indices .+= 1
	indptr .+= 1

	_fix_sparse_buffers!(P,N,indptr,indices,data)

	matrix = SparseMatrixCSC(P, N, indptr, indices, data)
	trans ? copy(matrix') : matrix
end

function _readh5ad_dense(::Type{T}, g) where T
	_read_h5_attribute(g, "encoding-version", "0.2.0") == "0.2.0" || @warn "Expected array encoding-version to be \"0.2.0\""
	_try_convert_array(T, read(g))
end

# read data with encoding-type "csr_matrix", "csc_matrix", or "array"
function _readh5ad_array(T, g)
	t = _read_h5_attribute(g, "encoding-type")

	# NB: CSR in Python is CSC in Julia, (Python is row major and Julia is column major)
	t == "csr_matrix" && return _readh5ad_sparse(T, g, false)
	t == "csc_matrix" && return _readh5ad_sparse(T, g, true)

	t == "array" && return _readh5ad_dense(T, g)

	error("Unknown matrix encoding-type: $t")
end
_readh5ad_array(g) = _readh5ad_array(Any, g)



function _anndata_version(h5)
	t = _read_h5_attribute(h5, "encoding-type", "anndata")
	@assert t == "anndata" "Root has unknown encoding-type"
	VersionNumber(_read_h5_attribute(h5, "encoding-version", "0.0.0"))
end


function _default_matrix_paths(h5)
	# TODO: should this depend on anndata_version?
	p = ["raw/X", "layers/sparse", "X"]
	if haskey(h5, "layers")
		for layer in keys(h5["layers"])
			layer === "sparse" && continue # already handled
			push!(p, "layers/$layer")
		end
	end
	p
end


"""
	loadh5ad([T=Int], filename; var_id_column=:id, obs_id_column=:cell_id, [matrix_path])

Experimental loading of .h5ad files.

Arguments:
* `T` (optional) - Element type of matrix.
* `filename` - Path of file to load.

Keyword arguments:
* `var_id_column` - Name of var ID column.
* `obs_id_column` - Name of obs ID column.
* `matrix_path` - Path inside the h5ad file deciding which matrix to load. Can be a vector, which means that multiple paths will be searched. The first matrix found which can be converted to the desired eltype `T` is returned. Defaults to searching in "raw/X", "layers/sparse", "X", and any other matrices in "layers".

"""
function loadh5ad(T, filename; obs_id_column=:cell_id, var_id_col=:id, matrix_path=nothing)
	h5open(filename) do h5
		anndata_version = _anndata_version(h5)
		anndata_version in (v"0.1.0", v"0.0.0") || @warn "Unrecognized anndata version: $anndata_version"

		# --- Variable annotations ---
		var = _readh5ad_dataframe(h5["var"]; id_column=var_id_col, anndata_version)

		# --- Cell annotations ---
		obs = _readh5ad_dataframe(h5["obs"]; id_column=obs_id_column, anndata_version)

		if matrix_path === nothing
			matrix_path = _default_matrix_paths(h5)
		elseif !(matrix_path isa AbstractArray)
			matrix_path = [matrix_path]
		end

		X = nothing
		for mp in matrix_path
			haskey(h5, mp) || continue
			X = _readh5ad_array(T, h5[mp])
			X !== nothing && break
		end
		X === nothing && error("No suitable matrix found: try to specify `matrix_path` or change the matrix eltype `T`.")

		@assert size(X,1) == size(var,1) "Size mismatch: matrix has size $(size(X)), but var has $(size(var,1)) rows."
		@assert size(X,2) == size(obs,1) "Size mismatch: matrix has size $(size(X)), but obs has $(size(obs,1)) rows."

		DataMatrix(X, var, obs)
	end
end
loadh5ad(filename; kwargs...) = loadh5ad(Int, filename; kwargs...)
