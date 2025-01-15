# Intended to be public function in low-level API
function combine_var(var::Vector{DataFrame};
                     prefilter = "feature_type"=>isequal("Gene Expression"),
                     extra_id_cols = "feature_type")
	isempty(var) && return DataFrame()
	
	id_col_names = only.(names.(var, 1))
	@assert all(isequal(id_col_names[1]), id_col_names) "Variable ID columns must match"

	id_cols = vcat(id_col_names[1], extra_id_cols)

	if filter !== nothing
		var = filter.(prefilter, var)
	end

	# Consider trying to create a merged list of vars that respect the order of all samples (when possible).
	# We can adapt Kahn's algorithm to do this.

	c = coalesce.(vcat(var..., cols=:union),"")
	g = groupby(c, id_cols)

	# Bug in DataFrames? if valuecols(g) is empty, an empty DataFrame is returned, but we want the keys.
	# combine(g, valuecols(g) .=> _value_or_ambiguous; renamecols=false)

	# Workaround.
	all_cols = vcat(Symbol.(id_cols), valuecols(g))
	combine(g, all_cols .=> _value_or_ambiguous; renamecols=false)
end

# WIP.
# Later intended to be public function in low-level API.
function sample_var_indices(sample_var::DataFrame, var::DataFrame;
                            extra_id_cols = "feature_type")
	@assert names(sample_var,1) == names(var,1) "Variable ID columns must match"
	id_cols = vcat(names(var,1), extra_id_cols)

	# Should return a vector of indices as long as `var`, telling where in `sample_var` to find each row in `var`.
	# Missing rows are allowed.
	table_indexin(var, sample_var; cols=id_cols)
end


# WIP.
# Later intended to be public function in low-level API.
function subset_by_var_indices(X::SparseMatrixCSC, var_ind::Vector{<:Union{Int,Nothing}}; reuse_memory=false)
	# Rows where `var_ind` is nothing, should be filled with zeros.
	# Otherwise pick corresponding row from `X`.

	P = length(var_ind)
	if size(X,1)==P && var_ind == 1:P
		return X # early out, nothing to do
	end

	# 1. Subset/order rows in X
	# 2. Manipulate/remap row indices to defacto insert rows of zeros
	# 3. Use new row indices to create a sparse matrix of the right size

	var_ind_matching = identity.(filter(!isnothing, var_ind)) # This removes Nothing from the eltype
	# Duplicates not allowed - That is, the same row in sample_var is not allowed to map to multiple rows in var.
	@assert allunique(var_ind_matching) "Each row in `sample_var` can match at most one row in `var`." # TODO: report ID of duplicates in error message

	if var_ind_matching != 1:length(var_ind_matching)
		X = X[var_ind_matching,:] # this is faster when var_ind_matching is sorted
	end

	row_remapper = identity.(indexin(var_ind, var_ind_matching)) # We know all are matching, so we can remove Nothing from eltype
	rv = getindex(Ref(row_remapper), rowvals(X)) # This is fine because the relative order of rows is kept

	N = size(X,2)
	colptr = SparseArrays.getcolptr(X) # annoying to have to use internal function
	nz = nonzeros(X)
	if !reuse_memory
		colptr = copy(colptr)
		nz = copy(nz)
	end
	return SparseMatrixCSC(P, N, colptr, rv, nz)
end


# WIP
# Later intended to be public function in low-level API.
function load_sample_matrix(f, io, var_ind)
	X = f(io)
	subset_by_var_indices(X, var_ind; reuse_memory=true)
end
load_sample_matrix(io, var_ind) = load_sample_matrix(read10x_matrix, io, var_ind)
