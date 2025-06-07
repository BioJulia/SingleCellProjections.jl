# WIP.
# Intended to be public function in low-level API.
function combine_obs(obs::Vector{DataFrame}, sample_names::Vector{String};
                     id_col = "cell_id", id_delim = '_',
                     sample_name_col = "sample_name")
	@assert length(obs) == length(sample_names)

	id_col_names = only.(names.(obs, 1))
	@assert all(isequal(id_col_names[1]), id_col_names) "Observation ID columns must match"

	old_id_col = id_col_names[1]

	obs = copy.(obs; copycols=false) # share vectors, but make it possible to add columns
	for (o,sn) in zip(obs, sample_names)
		ids = string.(sn, id_delim, o[!,1])

		if id_col == old_id_col
			o[!,1] = ids # NB: [!,1] ensures we don't change the original column, but store a new vector just reusing the column name
		else
			insertcols!(o, 1, id_col=>ids)
		end

		if sample_name_col !== nothing
			insertcols!(o, 2, sample_name_col=>sn)
		end
	end
	vcat(obs..., cols=:union)
end

# WIP.
# Intended to be public function in low-level API.
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
function subset_by_var_indices(X::SparseMatrixCSC{Tv,Ti},
                               var_ind::Vector{<:Union{Integer,Nothing}};
                               reuse_memory=false) where {Tv,Ti}
	# Rows where `var_ind` is nothing, should be filled with zeros.
	# Otherwise pick corresponding row from `X`.

	P = length(var_ind)
	if size(X,1)==P && var_ind == 1:P
		return X # early out, nothing to do
	end

	# 1. Subset/order rows in X
	# 2. Manipulate/remap row indices to defacto insert rows of zeros
	# 3. Use new row indices to create a sparse matrix of the right size

	var_ind_matching = something.(filter(!isnothing, var_ind)) # remove `Nothing` from eltype (and error if `nothing` is encountered)
	# Duplicates not allowed - That is, the same row in sample_var is not allowed to map to multiple rows in var.
	@assert allunique(var_ind_matching) "Each row in `sample_var` can match at most one row in `var`." # TODO: report ID of duplicates in error message

	if var_ind_matching != 1:size(X,1)
		X = X[var_ind_matching,:] # this is faster when var_ind_matching is sorted
	end

	# We know all are matching, so we can remove Nothing from eltype.
	# Also convert eltype to Ti to ensure output SparseMatrixCSC matches input SparseMatrixCSC.
	row_remapper = convert(Vector{Ti}, indexin(var_ind, var_ind_matching))
	rv = getindex.(Ref(row_remapper), rowvals(X)) # This is fine because the relative order of rows is kept

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
# This function mostly makes sense in a Spec workflow, where the result of this is cached.
function load_sample_matrix_metadata(args...)
	# This could be optimized, by not actually performing the subsetting, but probably not worth it.
	# We could also shortcut to use read10x_metadata if variables are kept as is.
	X = load_sample_matrix(args...)
	(size(X,1), size(X,2), nnz(X))
end

# Probably find a nicer way?
read10x_matrix_int_int32(io) = read10x_matrix(io, SparseMatrixCSC{Int,Int32})


# WIP
# Later intended to be public function in low-level API.
# f is function that loads raw sample matrix
# io is filename or IO
function load_sample_matrix(f, io, var_ind::Vector{<:Union{Int,Nothing}})
	X = f(io)
	subset_by_var_indices(X, var_ind; reuse_memory=true)
end
load_sample_matrix(io, var_ind) = load_sample_matrix(read10x_matrix_int_int32, io, var_ind)


# WIP
# Later intended to be public function in low-level API.
# fs:               Function (or vector of functions) invoked to load matrix
# ios:              Vector of filenames/IO objects used to load matrix
# matrix_metadatas: Vector of (P,N,nnz) for each sample_matrix (after `subset_by_var_indices` has been applied)
# var_inds:         Vector of var_ind for each matrix
function load_hcat_sample_matrices(fs, ::Type{Tv}, ::Type{Ti},
                                   ios::Vector, matrix_metadatas::Vector{Tuple{Int,Int,Int}},
                                   var_inds::Vector{<:Vector{<:Union{Int,Nothing}}}) where {Tv,Ti}
	n_samples = length(ios)
	@assert n_samples >= 1
	@assert length(matrix_metadatas) == n_samples
	@assert length(var_inds) == n_samples
	if fs isa AbstractVector
		@assert length(fs) == n_samples
	else
		fs = Iterators.repeated(fs, n_samples)
	end

	P = length(var_inds[1])
	@assert all(x->length(x)==P, var_inds) "Number of variables do not match between sample matrices."

	N = sum(x->x[2], matrix_metadatas)
	nnonzeros = sum(x->x[3], matrix_metadatas)

	colptr = Vector{Ti}(undef, N+1)
	rowval = Vector{Ti}(undef, nnonzeros)
	nzval  = Vector{Tv}(undef, nnonzeros)

	colptr_offset = 0
	nnz_offset = 0

	for (f,io,(P_sample,N_sample,nnz_sample),var_ind) in zip(fs,ios,matrix_metadatas,var_inds)
		@assert P_sample == P
		X = load_sample_matrix(f, io, var_ind) # load matrix adapted to var_ind
		X::SparseMatrixCSC{Tv,Ti}
		@assert nnz(X) == nnz_sample
		@assert size(X,1) == P_sample
		@assert size(X,2) == N_sample

		cp = SparseArrays.getcolptr(X)
		colptr[colptr_offset .+ (1:N_sample+1)] .= cp .+ nnz_offset
		rowval[nnz_offset .+ (1:nnz_sample)] .= rowvals(X)
		nzval[nnz_offset .+ (1:nnz_sample)] .= nonzeros(X)

		colptr_offset += N_sample
		nnz_offset += nnz_sample
	end

	SparseMatrixCSC(P, N, colptr, rowval, nzval)
end
load_hcat_sample_matrices(fs, ios, matrix_metadatas, var_inds) =
	load_hcat_sample_matrices(fs, Int, Int32, ios, matrix_metadatas, var_inds)
load_hcat_sample_matrices(Tv::DataType, Ti::DataType, ios, matrix_metadatas, var_inds) =
	load_hcat_sample_matrices(read10x_matrix_int_int32, Tv, Ti, ios, matrix_metadatas, var_inds)
load_hcat_sample_matrices(ios, matrix_metadatas, var_inds) =
	load_hcat_sample_matrices(read10x_matrix_int_int32, ios, matrix_metadatas, var_inds)
