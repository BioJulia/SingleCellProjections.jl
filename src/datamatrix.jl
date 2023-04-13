function _detect_var_id_cols(var)
	id_cols = String[]
	push!(id_cols, hasproperty(var, "id") ? "id" : first(names(var)))
	hasproperty(var, :feature_type) && push!(id_cols, "feature_type")
	@assert allunique(id_cols)

	v = select(var, id_cols; copycols=false)
	@assert size(unique(v),1) == size(v,1) "Failed to autodetect unique variable IDs (tried $id_cols)."
	id_cols
end

function _detect_obs_id_cols(obs)
	id_col = hasproperty(obs, "id") ? "id" : first(names(obs))
	@assert allunique(obs[!,id_col]) "Failed to autodetect unique observation IDs (tried \"$id_col\")."
	String[id_col]
end


"""
	struct DataMatrix{T,Tv,To}

A `DataMatrix` represents a matrix together with annotations for variables and observations.

Fields:
* `matrix::T` - The matrix.
* `var::Tv` - Variable annotations.
* `obs::To` - Observation annotations.
* `var_id_cols::Vector{String}` - Which column(s) to use as IDs.
* `obs_id_cols::Vector{String}` - Which column(s) to use as IDs.
* `models::Vector{ProjectionModel}` - Models used in the creation of this `DataMatrix`.

The rows of the `var` and `obs` tables must be unique, considering only the `var_id_cols`/`obs_id_cols`.
"""
struct DataMatrix{T,Tv,To}
	matrix::T
	var::Tv
	obs::To
	var_id_cols::Vector{String}
	obs_id_cols::Vector{String}
	models::Vector{ProjectionModel}
	function DataMatrix(matrix::T, var::Tv, obs::To, var_id_cols, obs_id_cols, models) where {T,Tv,To}
		P,N = size(matrix)
		p = size(var,1)
		n = size(obs,1)
		if P != p || N != n
			throw(DimensionMismatch(string("Matrix has dimensions (",P,',',N,"), but there are ", p, " variable annotations and ", n, " observation annotations.")))
		end

		# NB: copy because we do not want different DataMatrices to share the same storage
		var_id_cols = var_id_cols !== nothing ? copy(var_id_cols) : _detect_var_id_cols(var)
		obs_id_cols = obs_id_cols !== nothing ? copy(obs_id_cols) : _detect_obs_id_cols(obs)

		table_validatecols(var, var_id_cols)
		table_validatecols(obs, obs_id_cols)
		table_validateunique(var, var_id_cols)
		table_validateunique(obs, obs_id_cols)
		new{T,Tv,To}(matrix, var, obs, var_id_cols, obs_id_cols, models)
	end
end


"""
	DataMatrix(matrix, var, obs; var_id_cols=nothing, obs_id_cols=nothing)

Create a `DataMatrix` with the given `matrix`, `var` and `obs`.

Columns to use for `var`/`obs` IDs can be explicitly set with `var_id_cols`/`obs_id_cols`.
Otherwise, an attempt will be made to autodetect the ID columns.
"""
DataMatrix(matrix, var, obs; var_id_cols=nothing, obs_id_cols=nothing) =
	DataMatrix(matrix, var, obs, var_id_cols, obs_id_cols, ProjectionModel[])


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
	set_var_id_cols!(data::DataMatrix, var_id_cols::Vector{String})

Set which column(s) to use as variable IDs.
The rows of the `data.var` table must be unique, considering only the `var_id_cols` columns.
"""
function set_var_id_cols!(data::DataMatrix, var_id_cols::Vector{String})
	table_validatecols(data.var, var_id_cols)
	table_validateunique(data.var, var_id_cols)
	copy!(data.var_id_cols, var_id_cols)
end

"""
	set_obs_id_cols!(data::DataMatrix, obs_id_cols::Vector{String})

Set which column(s) to use as observation IDs.
The rows of the `data.obs` table must be unique, considering only the `obs_id_cols` columns.
"""
function set_obs_id_cols!(data::DataMatrix, obs_id_cols::Vector{String})
	table_validatecols(data.obs, obs_id_cols)
	table_validateunique(data.obs, obs_id_cols)
	copy!(data.obs_id_cols, obs_id_cols)
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
	project(data::DataMatrix, model::ProjectionModel, args...; verbose=true, kwargs...)

Convenience function for projection onto multiple models. Essentially calls `foldl` and prints some `@info` messages (if `verbose=true`).
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
	              obs::Union{Symbol,String,DataFrame} = "",
	              var_id_cols,
	              obs_id_cols)

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
* `var_id_cols` - New ID columns. Defaults to the same as data, or "id" if new variables were generated using the "prefix" above.
* `obs_id_cols` - See `var_id_cols`.

"""
function update_matrix(data::DataMatrix, matrix, model=nothing;
                       var::Union{Symbol,String,DataFrame} = "",
                       obs::Union{Symbol,String,DataFrame} = "",
                       var_id_cols = var isa String ? ["id"] : data.var_id_cols,
                       obs_id_cols = obs isa String ? ["id"] : data.obs_id_cols)
	models = model !== nothing ? vcat(data.models,model) : ProjectionModel[]
	var = _update_annot(data.var, var, size(matrix,1))
	obs = _update_annot(data.obs, obs, size(matrix,2))
	DataMatrix(matrix, var, obs, var_id_cols, obs_id_cols, models)
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
function _showannotations(io, annotations, id_cols, header)
	n = names(annotations)
	print(io, header, ": ")
	actual_id_cols = intersect(n, id_cols)
	bad_id_cols = setdiff(id_cols, actual_id_cols)
	other_cols = setdiff(n, actual_id_cols)

	c = 0
	for name in actual_id_cols
		_printannotation(io, name, c>0; underline=true)
		c += 1
	end
	for name in bad_id_cols
		_printannotation(io, name, c>0; underline=true, reverse=true)
		c += 1
	end
	for name in other_cols
		c>=10 && (print(io, ", ..."); break)
		_printannotation(io, name, c>0)
		c += 1
	end
end
_showvar(io, data) = _showannotations(io, data.var, data.var_id_cols, "Variables")
_showobs(io, data) = _showannotations(io, data.obs, data.obs_id_cols, "Observations")

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
