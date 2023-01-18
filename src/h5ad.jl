

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


function _readh5ad_dataframe_column(g)
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


function _readh5ad_dataframe(g; id_column)
	@assert read(attributes(g),"encoding-type") == "dataframe"

	_index = read(attributes(g),"_index")
	id = _readh5ad_dataframe_column(g[_index])

	df = DataFrame(id_column=>id)

	
	column_order = read(attributes(g),"column-order")
	if !isempty(column_order)
		columns = [_readh5ad_dataframe_column(g[c]) for c in column_order]

		df2 = DataFrame(column_order .=> columns)
		df = hcat(df,df2)
	end
	df
end



"""
	loadh5ad(filename; var_id_column=:id, obs_id_column=:id)

Experimental loading of .h5ad files.
"""
function loadh5ad(filename; obs_id_column=:id, var_id_col=:id)
	h5open(filename) do h5
		@assert read(attributes(h5), "encoding-type") == "anndata"


		# --- Cell annotations ---
		obs = _readh5ad_dataframe(h5["obs"]; id_column=obs_id_column)

		# --- Variable annotations ---
		var = _readh5ad_dataframe(h5["var"]; id_column=var_id_col)


		# --- Count Matrix ---
		X = h5["X"]
		@assert read(attributes(X), "encoding-type") == "csr_matrix"
		
		shape = read(attributes(X), "shape")
		@assert length(shape)==2
		N,P = shape

		data = convert(Vector{Int}, read(X, "data")) # TODO: maybe accept floats later?
		indices = convert(Vector{Int32}, read(X,"indices"))
		indptr = convert(Vector{Int32}, read(X,"indptr"))

		indices .+= 1
		indptr .+= 1

		matrix = SparseMatrixCSC(P, N, indptr, indices, data)


		DataMatrix(matrix, var, obs)
	end
end

