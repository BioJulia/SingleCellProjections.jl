function samplenamesfromfilenames(filenames::AbstractVector)::Union{Vector{String},Nothing}
	length(filenames) == 1 && return nothing

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

samplenamesfromfilenames(::Any) = nothing


struct Lazy10xMatrix{Tv,Ti}
	sz::Tuple{Int,Int}
	nnz::Int
	filename::String
end
Lazy10xMatrix(::Type{Tv},::Type{Ti}, args...) where {Tv,Ti} = Lazy10xMatrix{Tv,Ti}(args...)

Base.:(==)(a::Lazy10xMatrix, b::Lazy10xMatrix) = false
function Base.:(==)(a::Lazy10xMatrix{T,Tv}, b::Lazy10xMatrix{T,Tv}) where {T,Tv}
	all(i->getfield(a,i)==getfield(b,i), 1:nfields(a))
end


struct LazyMergedMatrix{Tv,Ti}
	sz::Tuple{Int,Int}
	nnz::Int
	data::Vector{DataMatrix}
	var_id_cols::Vector{String}
end
LazyMergedMatrix(::Type{Tv},::Type{Ti}, args...) where {Tv,Ti} = LazyMergedMatrix{Tv,Ti}(args...)

Base.:(==)(a::LazyMergedMatrix, b::LazyMergedMatrix) = false
function Base.:(==)(a::LazyMergedMatrix{T,Tv}, b::LazyMergedMatrix{T,Tv}) where {T,Tv}
	all(i->getfield(a,i)==getfield(b,i), 1:nfields(a))
end

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


load_counts(data::DataMatrix; callback=nothing, duplicate_var, duplicate_obs) = data

"""
	load_counts(data::DataMatrix{<:Lazy10xMatrix})

Load counts for a lazily loaded 10x DataMatrix.

See also: [`load10x`](@ref)
"""
load_counts(data::DataMatrix{<:Lazy10xMatrix}; callback=nothing, kwargs...) = DataMatrix(getmatrix(data.matrix), data.var, data.obs, kwargs...)

"""
	load_counts(data::DataMatrix{<:LazyMergedMatrix})

Merge/load counts for a lazily merged DataMatrix.
"""
function load_counts(data::DataMatrix{<:LazyMergedMatrix{Tv,Ti}}; callback=nothing, kwargs...) where {Tv,Ti}
	lazy_matrix = data.matrix

	sample_features = getfield.(lazy_matrix.data,:var)
	matrices = getfield.(lazy_matrix.data,:matrix)
	matrix = _merge_matrices(Tv, Ti, data.var, sample_features, matrices; lazy_matrix.var_id_cols, callback)
	matrix===nothing && return nothing
	update_matrix(data, matrix; var=:keep, obs=:keep, kwargs...)
end



function _load10x_metadata(io)
	features = read10x_features(io, DataFrame)
	cells = read10x_barcodes(io, DataFrame)
	P,N,nz = read10x_matrix_metadata(io)
	P,N,nz,features,cells
end

"""
	load10x(filename; lazy=false, var_id=nothing, var_id_delim='_')

Load a CellRanger ".h5" or ".mtx[.gz]" file as a DataMatrix.

* `lazy` - If `true`, the count matrix itself will not be loaded, only features and barcodes. This is used internally in `load_counts` to merge samples more efficiently. Use `load_counts` to later load the count data.
* `var_id` - If a pair `var_id_col=>cols`, the contents of columns `cols` will be merged to create new IDs. Useful to ensure that IDs are unique.
* `var_id_delim` - Delimiter used to when merging variable columns to create the variable id column.

# Examples
Load counts from a CellRanger ".h5" file. (Recommended.)
```julia
julia> counts = load10x("filtered_feature_bc_matrix.h5")
```

Load counts from a CellRanger ".mtx" file. Tries to find barcode and feature annotation files in the same folder.
```julia
julia> counts = load10x("matrix.mtx.gz")
```

Lazy loading followed by loading.
```julia
julia> counts = load10x("filtered_feature_bc_matrix.h5");
julia> counts = load_counts(counts)
```

See also: [`load_counts`](@ref)
"""
function load10x(filename; lazy=false, var_id::Union{Nothing,Pair{String,<:Any}}=nothing, var_id_delim='_', kwargs...)
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

	if var_id !== nothing
		var_id_col,cols = var_id
		var_ids = join.(values.(eachrow(select(features,cols))), var_id_delim)

		var_id_col in names(features) && select!(features, Not(var_id_col)) # remove ID column if existing so we can insert at position 1 below
		insertcols!(features, 1, var_id_col=>var_ids)
	end

	DataMatrix(matrix, features, cells; kwargs...)
end


"""
	load_counts([loadfun=load10x], filenames;
                sample_names,
                sample_name_col,
                obs_id_col = "cell_id",
                lazy,
                lazy_merge = false,
                obs_id_delim = '_',
                obs_id_prefixes = sample_names,
                extra_var_id_cols::Union{Nothing,String,Vector{String}},
                duplicate_var,
                duplicate_obs,
                callback=nothing)

Load and merge multiple samples efficiently.

Defaults to loading 10x CellRanger files.
The files are first loaded lazily, then the merged count matrix is allocated and finally each sample is loaded directly into the merged count matrix.
(This strategy greatly reduces memory usage, since only one copy of data is needed instead of two.)

`filenames` specifies which files to load. (It can be a vector of filenames or a single filename string.)
For each file, `loadfun` is called.

* `sample_names` - Specify the sample names. Should be a vector of the same length as `filenames`. Set to `nothing` to not create a sample name annotation.
* `sample_name_col` - Column for sample names in `obs`, defaults to "sampleName".
* `obs_id_col` - Colum for merged `id`s in `obs`.
* `lazy` - Enable lazy loading. Defaults to true if `load10x` is used, and `false` otherwise.
* `lazy_merge` - Enable lazy merging, i.e. `var` and `obs` are created, but the count matrix merging is postponed until a second call to `load_counts`.
* `obs_id_delim` - Delimiter used when creating merged `obs` IDs.
* `obs_id_prefixes` - Prefix (one per sample) used to create new IDs. Set to nothing to keep old IDs. Defaults to `sample_names`.
* `extra_var_id_cols` - Additional columns to use to ensure variable IDs are unique during merging. Defaults to "feature_type" if that column is present for all samples. Can be a `Vector{String}` to include multiple columns. Set to nothing to disable.
* `duplicate_var` - Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate var IDs are found.
* `duplicate_obs` - Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate obs IDs are found.
* `callback` - Experimental callback functionality. The callback function is called between samples during merging. Return `true` to abort loading and `false` to continue.
* Additional kwargs (including `duplicate_var`/`duplicate_obs` if specified) are passed to `loadfun`.

# Examples

Load and name samples:
```julia
julia> counts = load_counts(["s1.h5", "s2.h5"]; sample_names=["Sample A", "Sample B"])
```

See also: [`load10x`](@ref), [`merge_counts`](@ref)
"""
function load_counts(loadfun, filenames;
                     sample_names=samplenamesfromfilenames(filenames),
                     sample_name_col = sample_names===nothing ? nothing : "sampleName",
                     lazy=loadfun==load10x,
                     lazy_merge=false,
                     obs_id_col = "cell_id",
                     obs_id_delim = '_',
                     obs_id_prefixes = sample_names,
                     extra_var_id_cols = :auto,
                     duplicate_var = nothing,
                     duplicate_obs = nothing,
                     callback=nothing,
                     kwargs...)

	filenames isa AbstractVector || (filenames = [filenames])
	sample_names isa Union{Nothing,AbstractVector} || (sample_names = [sample_names])

	# TODO: call callback between sample loads(?)
	args = lazy ? (;lazy) : (;)
	kwargs_var = duplicate_var !== nothing ? (;duplicate_var) : (;)
	kwargs_obs = duplicate_obs !== nothing ? (;duplicate_obs) : (;)

	samples = loadfun.(filenames; args..., kwargs_var..., kwargs_obs..., kwargs...) # Do *not* pass kwargs to loadfuns that might not support them

	merge_counts(samples, sample_names; lazy=lazy_merge, sample_name_col, obs_id_col, obs_id_delim, obs_id_prefixes, extra_var_id_cols, callback, kwargs_var..., kwargs_obs...)
end

# default to 10x
load_counts(filenames; kwargs...) = load_counts(load10x, filenames; kwargs...)



_value_or_ambiguous(x) = Ref(length(unique(x))!=1 ? "ambiguous" : first(x))


function _merge_features(features, var_id_cols)
	id_col_names = only.(names.(features, 1))
	@assert all(==(first(id_col_names)), id_col_names) "Variable ID columns must match"

	# Consider trying to create a merged list of features that respect the order of all samples (when possible).

	c = coalesce.(vcat(features..., cols=:union),"")
	g = groupby(c, var_id_cols)

	# Bug in DataFrames? if valuecols(g) is empty, an empty DataFrame is returned, but we want the keys.
	# combine(g, valuecols(g) .=> _value_or_ambiguous; renamecols=false)

	# Workaround.
	cols = vcat(Symbol.(var_id_cols), valuecols(g))
	combine(g, cols .=> _value_or_ambiguous; renamecols=false)
end


function _merge_cells(samples, sample_names; sample_name_col, obs_id_col, obs_id_delim, obs_id_prefixes)
	id_col_names = only.(names.(getfield.(samples,:obs), 1))
	@assert all(==(first(id_col_names)), id_col_names) "Observation ID columns must match"

	old_obs_id_col = first(id_col_names)
	obs_id_col = @something obs_id_col old_obs_id_col


	sample_cells = DataFrame[]
	for (k,s) in enumerate(samples)
		c = copy(s.obs; copycols=false) # share vectors, but make it possible to add columns

		obs_ids = c[!,1]
		if obs_id_prefixes !== nothing
			obs_ids = string.(obs_id_prefixes[k], obs_id_delim, obs_ids)
		end

		if obs_id_col == old_obs_id_col # reuse old id column?
			c[!,1] = obs_ids # NB: [!,1] ensures we don't change the original column, but store a new vector just reusing the column name
		else
			insertcols!(c, 1, obs_id_col=>obs_ids)
		end

		sample_name_col !== nothing && insertcols!(c, 2, sample_name_col=>sample_names[k])

		push!(sample_cells, c)
	end

	cells = vcat(sample_cells...; cols=:union)

	size(unique(select(cells,1)),1) != size(cells,1) && error("Merged cell ids are not unique.")

	cells
end


function _insert_matrix!(colptr, rowval, nzval, colptr_offset, nnz_offset, feature_ind, sf, var_id_cols, A)
	cp = SparseArrays.getcolptr(A)
	rv = rowvals(A)
	nz = nonzeros(A)

	colptr[colptr_offset .+ (1:size(A,2)+1)] .= cp .+ nnz_offset

	# Figure out how to map feature indices in sample to global feature indices
	sf_ind = select(sf, var_id_cols)
	leftjoin!(sf_ind, feature_ind; on=var_id_cols)
	row_ind = identity.(sf_ind.__row__) # identity narrows type to Int (i.e. exludes Missing)

	if issorted(row_ind)
		rowval[nnz_offset .+ (1:nnz(A))] .= getindex.(Ref(row_ind),rv)
		nzval[nnz_offset .+ (1:nnz(A))] .= nz
	else
		# rowvals might be out of order after remapping, sort within each column
		# TODO: avoid allocations below by using scratch spaces?
		for j in 1:size(A,2)
			rng = cp[j]:cp[j+1]-1
			rv_j = row_ind[rv[rng]] # remapped row indices, not guaranteed to be sorted
			nz_j = @view nz[rng]
			perm = sortperm(rv_j)
			rowval[rng.+nnz_offset] .= rv_j[perm]
			nzval[rng.+nnz_offset] .= nz_j[perm]
		end
	end
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


"""
	merge_counts(samples, sample_names;
	             lazy=false,
	             sample_name_col = sample_names===nothing ? nothing : "sampleName",
	             obs_id_col = "cell_id",
	             obs_id_delim = '_',
	             obs_id_prefixes = sample_names,
	             extra_var_id_cols::Union{Nothing,String,Vector{String}},
	             duplicate_var,
	             duplicate_obs,
	             callback=nothing)

Merge `samples` to create one large DataMatrix, by concatenating the `obs`.
The union of the variables from the samples is used, and if a variable is not present in a sample, the count will be set to zero.

The `obs` IDs are created by concatenating the current `obs` ID columns, together with the `sample_names` (if provided).

* `lazy` - Lazy merging. Use `load_counts` to actually perform the merging.
* `sample_name_col` - Column in which the `sample_names` are stored.
* `obs_id_col` - Name of `obs` ID column after merging. (Set to nothing to keep old column name.)
* `obs_id_delim` - Delimiter used when merging `obs` IDs.
* `obs_id_prefixes` - Prefix (one per sample) used to create new IDs. Set to nothing to keep old IDs. Defaults to `sample_names`.
* `extra_var_id_cols` - Additional columns to use to ensure variable IDs are unique during merging. Defaults to "feature_type" if that column is present for all samples. Can be a `Vector{String}` to include multiple columns. Set to nothing to disable.
* `duplicate_var` - Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate var IDs are found.
* `duplicate_obs` - Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate obs IDs are found.
* `callback` - Experimental callback functionality. The callback function is called between samples during merging. Return `true` to abort loading and `false` to continue.

See also: [`load_counts`](@ref)
"""
function merge_counts(samples, sample_names;
                      lazy=false,
                      sample_name_col = sample_names===nothing ? nothing : "sampleName",
                      obs_id_col = "cell_id",
                      obs_id_delim = '_',
                      obs_id_prefixes = sample_names,
                      extra_var_id_cols = :auto,
                      duplicate_var = nothing,
                      duplicate_obs = nothing,
                      callback=nothing)
	@assert sample_name_col===nothing || length(samples)==length(sample_names)

	sample_features = getfield.(samples,:var)

	if extra_var_id_cols === nothing
		extra_var_id_cols = String[]
	elseif extra_var_id_cols == :auto
		extra_var_id_cols = all(x->hasproperty(x, "feature_type"), sample_features) ? ["feature_type"] : String[]
	else
		extra_var_id_cols isa AbstractVector || (extra_var_id_cols = [extra_var_id_cols])
	end
	var_id_cols = vcat(names(first(sample_features),1), extra_var_id_cols)

	features = _merge_features(sample_features, var_id_cols)
	cells = _merge_cells(samples, sample_names; sample_name_col, obs_id_col, obs_id_delim, obs_id_prefixes)

	matrices = getfield.(samples,:matrix)
	Tv = eltype(first(matrices))
	Ti = _indextype(first(matrices))

	P = size(features,1)
	N = sum(A->size(A,2), matrices)
	nnonzeros = sum(nnz, matrices)

	lazy_matrix = LazyMergedMatrix(Tv, Ti, (P,N), nnonzeros, convert(Vector{DataMatrix},samples), var_id_cols)

	kwargs_var = duplicate_var !== nothing ? (;duplicate_var) : (;)
	kwargs_obs = duplicate_obs !== nothing ? (;duplicate_obs) : (;)
	lazy_data = DataMatrix(lazy_matrix, features, cells; kwargs_var..., kwargs_obs...)
	lazy ? lazy_data : load_counts(lazy_data; callback, kwargs_var..., kwargs_obs...)
end
