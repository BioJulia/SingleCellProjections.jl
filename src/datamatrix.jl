function _get_nonunique(table, col)
	bad_ind = findfirst(nonunique(table, col))
	bad_ind !== nothing ? table[bad_ind,col] : nothing
end

function _validateunique(table, col, report, suffix)
	report == :ignore && return
	@assert report in (:warn, :error)
	bad_id = _get_nonunique(table,col)
	msg = string("ID \"", bad_id, "\" is not unique.", suffix)
	report == :error && error(msg)
	report == :warn && @warn(msg)
	return
end

validateunique_var(table, col, report) = _validateunique(table,col,report," Use duplicate_var=x, where x is :error, :warn or :ignore to control behavior.")
validateunique_obs(table, col, report) = _validateunique(table,col,report," Use duplicate_obs=x, where x is :error, :warn or :ignore to control behavior.")

"""
	struct DataMatrix{T,Tv,To}

A `DataMatrix` represents a matrix together with annotations for variables and observations.

Fields:
* `matrix::T` - The matrix.
* `var::Tv` - Variable annotations.
* `obs::To` - Observation annotations.
* `models::Vector{ProjectionModel}` - Models used in the creation of this `DataMatrix`.

The rows of the `var` and `obs` tables must be unique, considering only the `var_id_cols`/`obs_id_cols`.
"""
struct DataMatrix{T,Tv,To}
	matrix::T
	var::Tv
	obs::To
	models::Vector{ProjectionModel}
	function DataMatrix(matrix::T, var::Tv, obs::To, models; duplicate_var=:warn, duplicate_obs=:warn) where {T,Tv,To}
		P,N = size(matrix)
		p = size(var,1)
		n = size(obs,1)
		if P != p || N != n
			throw(DimensionMismatch(string("Matrix has dimensions (",P,',',N,"), but there are ", p, " variable annotations and ", n, " observation annotations.")))
		end

		validateunique_var(var, 1; report=duplicate_var)
		validateunique_var(obs, 1; report=duplicate_obs)
		new{T,Tv,To}(matrix, var, obs, models)
	end
end


"""
	DataMatrix(matrix, var, obs; kwargs...)

Create a `DataMatrix` with the given `matrix`, `var` and `obs`.

The first column of `var`/`obs` are used as IDs.

Kwargs:
* `duplicate_var`: Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate var IDs are found.
* `duplicate_obs`: Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate obs IDs are found.
"""
DataMatrix(matrix, var, obs; kwargs...) =
	DataMatrix(matrix, var, obs, ProjectionModel[]; kwargs...)


"""
	DataMatrix()

Create an empty DataMatrix{Matrix{Float64},DataFrame,DataFrame}.
"""
DataMatrix() = DataMatrix(zeros(0,0),DataFrame(id=String[]),DataFrame(id=String[]))

Base.:(==)(a::DataMatrix, b::DataMatrix) = false
function Base.:(==)(a::DataMatrix{T,Tv,To}, b::DataMatrix{T,Tv,To}) where {T,Tv,To}
	all(i->getfield(a,i)==getfield(b,i), 1:nfields(a))
end


Base.size(data::DataMatrix) = size(data.matrix)
Base.size(data::DataMatrix, dim::Integer) = size(data.matrix, dim)

Base.axes(data::DataMatrix, d::Integer) = axes(data)[d] # needed to make end work in getindex



"""
	copy(data::DataMatrix; var=:copy, obs=:copy, matrix=:keep)

Copy DataMatrix `data`. By default, `var` and `obs` annotations are copied, but the `matrix` is shared.
Set kwargs `var`, `obs` and `matrix` to `:keep`/`:copy` for fine grained control.
"""
function Base.copy(data::DataMatrix; var=:copy, obs=:copy, matrix=:keep)
	@assert var in (:copy,:keep)
	@assert obs in (:copy,:keep)
	@assert matrix in (:copy,:keep)

	X = matrix==:copy ? copy(data.matrix) : data.matrix
	v = var==:copy ? copy(data.var) : data.var
	o = obs==:copy ? copy(data.obs) : data.obs

	DataMatrix(X, v, o, copy(data.models))
end



"""
	set_var_id_col!(data::DataMatrix, var_id_col::String; duplicate_var=:error)

Set which column to use as variable IDs. It will be moved to the first column of `data.var`.
The rows of this column in `data.var` must be unique.

* `duplicate_var`: Set to :ignore, :warn or :error to decide what happens if duplicate IDs are found.
"""
function set_var_id_col!(data::DataMatrix, var_id_col::String; duplicate_var=:error)
	table_validatecols(data.var, var_id_col)
	validateunique_var(data.var, var_id_col; report=duplicate_var)
	select!(data.var, var_id_col, Not(var_id_col)) # move var_id_col first
end

"""
	set_obs_id_col!(data::DataMatrix, obs_id_col::String; duplicate_obs=:error)

Set which column to use as observation IDs. It will be moved to the first column of `data.obs`.
The rows of this column in `data.obs` must be unique.

* `duplicate_obs`: Set to :ignore, :warn or :error to decide what happens if duplicate IDs are found.
"""
function set_obs_id_col!(data::DataMatrix, obs_id_col::String; duplicate_obs=:error)
	table_validatecols(data.obs, obs_id_col)
	validateunique_var(data.obs, obs_id_col; report=duplicate_obs)
	select!(data.obs, obs_id_col, Not(obs_id_col)) # move obs_id_col first
end



"""
	var_coordinates(data::DataMatrix)

Returns a matrix with coordinates for the variables. Only available for DataMatrices that have a dual representation (e.g. SVD/PCA).

In the case of `SVD` (PCA), `var_coordinates` returns the principal components as unit vectors.
"""
function var_coordinates end

"""
	obs_coordinates(data::DataMatrix)

Returns a matrix with coordinates for the observations. Not available for all types of DataMatrices.
Mostly useful for data matrices after dimension reduction such as `svd` or `force_layout` has been applied.

In the case of `SVD` (PCA), `obs_coordinates` returns the principal components, scaled by the singular values.
This is a a good starting point for downstream analysis, since it is the optimal linear approximation of the original data for the given number of dimensions.
"""
function obs_coordinates end


var_coordinates(data::DataMatrix{<:Union{SVD,LowRank}}) = var_coordinates(data.matrix)
obs_coordinates(data::DataMatrix{<:Union{SVD,LowRank}}) = obs_coordinates(data.matrix)
obs_coordinates(data::DataMatrix{<:AbstractMatrix}) = data.matrix


Base.getindex(data::DataMatrix, I::Index, J::Index) = filter_matrix(I, J, data)


_startswith(a::AbstractVector, b::AbstractVector) = length(a)>=length(b) && view(a,1:length(b)) == b

function project_from(data::DataMatrix, base::DataMatrix, from::DataMatrix, args...; kwargs...)
	_startswith(base.models, from.models) || throw(ArgumentError("The \"from\" model history is not consistent with \"base\" model history."))
	models = @view base.models[length(from.models)+1:end]
	project(data, models, args...; kwargs...)
end
function project_from(data::DataMatrix, base::DataMatrix, ::Nothing, args...; kwargs...)
	ind = 0
	if !isempty(data.models)
		last_model = data.models[end]
		ind = findlast(projection_isequal(last_model), base.models)
		if ind === nothing
			throw(ArgumentError("Could not identify starting point for projection, no model matching $last_model found in the base DataMatrix."))
		end
	end
	project(data, (@view base.models[ind+1:end]), args...; kwargs...)
end


"""
	project(data::DataMatrix, models, args...; verbose=true, kwargs...)

Convenience function for projection onto multiple `models`. Essentially calls `foldl` and prints some `@info` messages (if `verbose=true`).
In most cases, it is better to call `project(data, base::DataMatrix)` instead of using this method directly.
"""
function project(data::DataMatrix, models::AbstractVector{<:ProjectionModel}, args...; verbose=true, kwargs...)
	foldl(models; init=data) do d,m
		verbose && @info "Projecting onto $m"
		project(d,m,args...; verbose, kwargs...)
	end
end


"""
	project(data::DataMatrix, model::ProjectionModel, args...; verbose=true, kwargs...)

Core projection function. Project `data` based on the single `ProjectionModel` `model`.
In most cases, it is better to call `project(data, base::DataMatrix)` instead of using this method directly.
"""
function project(data::DataMatrix, model::ProjectionModel, args...; verbose=true, kwargs...)
	model,kwargs = _update_model(model; kwargs...)
	project_impl(data, model, args...; verbose, kwargs...)
end



"""
	project(data::DataMatrix, base::DataMatrix, args...; from=nothing, kwargs...)

Project `data` onto `base`, by applying ProjectionModels from `base` one by one.

Since `data` already might have some models applied, `project` will try to figure out which models from `base` to use.
See "Examples" below for concrete examples. Here's a more technical overview:

Consider a `base` data matrix with four models:
```
base: A -> B -> C -> D
```

Given some new `data` (typically counts), we can project that onto `base`, given the result `proj` by applying all four models:
```
data:
proj: A -> B -> C -> D
```

If `data` already has some models applied (e.g. we already projected onto A and B above), `project` will look for the last model in `data` (in this case B) in the list of models in `base`, and only apply models after that (in this case C and D).
```
data: A -> B
proj: A -> B -> C -> D
```

It is also possible to use the `from` kwarg to specify exactly which models to apply.
(The models in `from` must be a prefix of the models in `base`, or in other words, `base` was created by applying additional operations to `from`.)
```
data: X
base: A -> B -> C -> D
from: A -> B
proj: X -> C -> D
```

Note that it is necessary to use the `from` kwarg if the last model in `data` does not occurr in `base`, because `project` cannot figure out on its own which models it makes sense to apply.


# Examples

First, we construct a "base" by loading counts, SCTransforming, normalizing, computing the svd and finally computing a force layout:
```julia
julia> fp = ["GSE164378_RNA_ADT_3P_P1.h5", "GSE164378_RNA_ADT_3P_P2.h5"];
julia> counts = load_counts(fp; sample_names=["P1","P2"]);
julia> transformed = sctransform(counts);
julia> normalized = normalize_matrix(transformed);
julia> reduced = svd(normalized; nsv=10);
julia> fl = force_layout(reduced; ndim=3, k=100)
  DataMatrix (3 variables and 35340 observations)
  Matrix{Float64}
  Variables: id
  Observations: id, sampleName, barcode
  Models: NearestNeighborModel(base="force_layout", k=10), SVD, Normalization, SCTransform
```
Note how the last line lists all `ProjectionModels` used in the creation of `fl`.

Next, let's load some more samples for projection:
```julia
julia> fp2 = ["GSE164378_RNA_ADT_3P_P5.h5", "GSE164378_RNA_ADT_3P_P6.h5"];
julia> counts2 = load_counts(fp2; sample_names=["P5","P6"]);
```

It is easy to project the newly loaded `counts2` onto the "base" force layout `fl`:
```julia
julia> project(counts2, fl)
DataMatrix (3 variables and 42553 observations)
  Matrix{Float64}
  Variables: id
  Observations: id, sampleName, barcode
  Models: NearestNeighborModel(base="force_layout", k=10), SVD, Normalization, SCTransform
```

We can also project in two or more steps, to get access to intermediate results:
```julia
julia> reduced2 = project(counts2, reduced)
DataMatrix (20239 variables and 42553 observations)
  SVD (10 dimensions)
  Variables: id, feature_type, name, genome, read, pattern, sequence, logGeneMean, outlier, beta0, ...
  Observations: id, sampleName, barcode
  Models: SVDModel(nsv=10), Normalization, SCTransform

julia> project(reduced2, fl)
DataMatrix (3 variables and 42553 observations)
  Matrix{Float64}
  Variables: id
  Observations: id, sampleName, barcode
  Models: NearestNeighborModel(base="force_layout", k=10), SVD, Normalization, SCTransform
```

If the DataMatrix we want to project is modified, we need to use the `from` kwarg to tell `project` which models to use:
```julia
julia> filtered = counts2[:,1:10:end]
DataMatrix (33766 variables and 4256 observations)
  SparseArrays.SparseMatrixCSC{Int64, Int32}
  Variables: id, feature_type, name, genome, read, pattern, sequence
  Observations: id, sampleName, barcode
  Models: FilterModel(:, 1:10:42551)

julia> reduced2b = project(filtered2, reduced; from=counts)
DataMatrix (20239 variables and 4256 observations)
  SVD (10 dimensions)
  Variables: id, feature_type, name, genome, read, pattern, sequence, logGeneMean, outlier, beta0, ...
  Observations: id, sampleName, barcode
  Models: SVDModel(nsv=10), Normalization, SCTransform, Filter
```

After that, it is possible to continue without specifying `from`:
```julia
julia> project(reduced2b, fl)
DataMatrix (3 variables and 4256 observations)
  Matrix{Float64}
  Variables: id
  Observations: id, sampleName, barcode
  Models: NearestNeighborModel(base="force_layout", k=10), SVD, Normalization, SCTransform, Filter
```
"""
project(data::DataMatrix, base::DataMatrix, args...; from=nothing, kwargs...) = project_from(data, base, from, args...; kwargs...)



function _update_annot(old, update::Symbol)
	@assert update in (:copy, :keep)
	update == :copy ? copy(old) : old
end
_update_annot(old, update::Symbol, ::Int) = _update_annot(old, update)

_update_annot(::Any, update::DataFrame, ::Int) = update

_update_annot(::Any, prefix::String, n::Int) = DataFrame(id=string.(prefix, 1:n))

"""
	update_matrix(data::DataMatrix, matrix, model=nothing;
	              var::Union{Symbol,String,DataFrame} = "",
	              obs::Union{Symbol,String,DataFrame} = "")

Create a new `DataMatrix` by replacing parts of `data` with new values.
Mostly useful when implementing new `ProjectionModel`s.

* `matrix` - the new matrix.
* `model` - will be appended to the list of models from `data`. If set to `nothing`, the resulting list of `models` will be empty.

Kwargs:
* `var` - One of:
  * `:copy` - Copy from `data`.
  * `:keep` - Share `var` with `data`.
  * `::DataFrame` - Replace with a new table with variable annotations.
  * `prefix::String` - Prefix, the new variables will be named prefix1, prefix2, etc.
* `obs` See `var`.

"""
function update_matrix(data::DataMatrix, matrix, model=nothing;
                       var::Union{Symbol,String,DataFrame} = "",
                       obs::Union{Symbol,String,DataFrame} = "")
	models = model !== nothing ? vcat(data.models,model) : ProjectionModel[]
	var = _update_annot(data.var, var, size(matrix,1))
	obs = _update_annot(data.obs, obs, size(matrix,2))
	DataMatrix(matrix, var, obs, models)
end




# - show -

function _showoverview(io, data)
	sz = size(data)
	print(io, "DataMatrix (", sz[1], " variables and ", sz[2], " observations)")
end

_showmatrix(io, matrix::T) where T = print(io, T)
_showmatrix(io, matrix::SVD) = print(io, "SVD (", innersize(matrix), " dimensions)")
_showmatrix(io, matrix::LowRank) = print(io, "LowRank (", innersize(matrix), " dimensions)")
_showmatrix(io, matrix::MatrixExpression) = show(io, matrix)

function _printannotation(io, name, show_delim; kwargs...)
	show_delim && print(io, ", ")
	printstyled(io, name; kwargs...)
end
function _showannotations(io, annotations, header)
	print(io, header, ": ")
	for (c,name) in enumerate(names(annotations))
		c>=10 && (print(io, ", ..."); break)
		_printannotation(io, name, c>1; underline=c==1, reverse=c==1 && !allunique(annotations[!,name])) # reverse if ID columns doesn't have unique values
	end
end
_showvar(io, data) = _showannotations(io, data.var, "Variables")
_showobs(io, data) = _showannotations(io, data.obs, "Observations")

function _showmodels(io, models)
	print(io, "Models: ")
	isempty(models) && return
	show(io, MIME"text/plain"(), models[end])
	length(models)==1 && return

	print(io, ", ")
	join(IOContext(io, :compact=>true), models[end-1:-1:max(1,end-5+1)], ", ")
	length(models)>5 && print(io, ", ...")
end


function Base.show(io::IO, ::MIME"text/plain", data::DataMatrix)
	_showoverview(io, data)
	println(io)
	print(io, "  ")
	_showmatrix(io, data.matrix)
	println(io)
	print(io, "  ")
	_showvar(io, data)
	println(io)
	print(io, "  ")
	_showobs(io, data)
	if !isempty(data.models)
		println(io)
		print(io, "  ")
		_showmodels(io, data.models)
	end
end
function Base.show(io::IO, data::DataMatrix)
	sz = size(data)
	print(io, sz[1], '×', sz[2], ' ')
	_showmatrix(io, data.matrix)
end
