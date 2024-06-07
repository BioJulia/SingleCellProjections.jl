_subsetmatrix(X::AbstractMatrix, I::Index, J::Index) = X[I,J]



function _gather_columns!(df, col::Union{Symbol,String}, external)
	a = find_annotation(col, external)
	a === nothing && throw(ArgumentError("External annotation \"$(cov.name)\" missing."))
	leftjoin!(df, a; on=names(obs,1))
	nothing
end

function _gather_columns!(df, cols::AbstractVector, external)
	for col in cols
		_gather_columns!(df, col, external)
	end
end


_filter_indices(::DataFrame, I::Index, ::Nothing=nothing) = I
_filter_indices(df::DataFrame, f, ::Nothing=nothing) = first(parentindices(filter(f,df;view=true)))
function _filter_indices(df::DataFrame, f::Pair{Union{Symbol,String,<:AbstractVector},Any}, external)
	df = select(df, 1; copycols=false) # just keep the ID column

	# 1. Find names of columns needed for predicate
	# 2. Extract columns needed for predicate (and join to IDs)
	_gather_columns!(df, first(f), external)

	# 3. Use standard _filter_indices
	_filter_indices(df, f)
end

struct FilterModel{Tv<:Index,To} <: ProjectionModel
	var_filter::Tv
	obs_filter::To
	var_match::DataFrame
	use_external_obs::Bool
	var::Symbol
	obs::Symbol

    function FilterModel(var_filter::Tv, obs_filter::To, var_match, use_external_obs, var, obs) where {Tv<:Index,To}
		# :keep only possible when indexing with :
		var_filter != Colon() && var == :keep && throw(ArgumentError("var = :keep is only allowed when indexing with :"))
		obs_filter != Colon() && obs == :keep && throw(ArgumentError("obs = :keep is only allowed when indexing with :"))
        new{Tv,To}(var_filter, obs_filter, var_match, use_external_obs, var, obs)
    end
end
function FilterModel(var_annots::Tv, var_filter, obs_filter; var=:copy, obs=:copy, external_var=nothing, use_external_obs=false) where Tv
	var_filter = _filter_indices(var_annots, var_filter, external_var)
	FilterModel(var_filter, obs_filter, select(var_annots, 1), use_external_obs, var, obs)
end
FilterModel(data::DataMatrix, args...; kwargs...) =
	FilterModel(data.var, args...; kwargs...)

function projection_isequal(m1::FilterModel, m2::FilterModel)
	m1.var_filter == m2.var_filter && m1.obs_filter == m2.obs_filter &&
	m1.var_match == m2.var_match && m1.use_external_obs == m2.use_external_obs
end


function update_model(m::FilterModel; var_filter=m.var_filter, obs_filter=nothing,
                      var=m.var, obs=m.obs, kwargs...)
	allow_obs_indexing = obs_filter !== nothing
	obs_filter === nothing && (obs_filter = m.obs_filter)
	model = FilterModel(var_filter, obs_filter, m.var_match, m.use_external_obs, var, obs)
	(model, (;allow_obs_indexing, kwargs...))
end


function _validate(var, model::FilterModel, allow_obs_indexing, ::Type{T}=typeof(model), name="obs_filter") where T
	if !allow_obs_indexing && !(model.obs_filter isa Colon) && model.obs_filter isa Index
		throw(ArgumentError("$(nameof(T)) has explicit observation indices. Use a different $(nameof(T)) or call project() with $name set."))
	end
end

_index_size(I::AbstractVector{<:Bool}) = count(I)
_index_size(I::AbstractVector) = length(I)


function _reordered_var_ind(I, var, var_match; verbose)
	# variables are not identical, we need to: reorder, ignore missing, get rid of extra
	var_ind = table_indexin(var_match, var; cols=names(var_match))

	n_removed = size(var,1) - count(!isnothing,var_ind)

	var_ind = var_ind[I] # remaining indices into data.var after filtering
	out_ind = var_ind[var_ind.!==nothing] # this is the updated `I` that can be used to index `var` above

	if verbose
		# - show info -
		n_missing = _index_size(I) - length(out_ind)
		n_removed>0 && @info "- Removed $n_removed variables that where not found in Model"
		n_missing>0 && @info "- Skipped $n_missing missing variables"
		issorted(out_ind) || @info "- Reordered variables to match Model"
	end

	out_ind
end

function project_impl(data::DataMatrix, model::FilterModel; external_obs=nothing, allow_obs_indexing=false, verbose=true)
	_validate(data.var, model, allow_obs_indexing)

	I = _filter_indices(data.var, model.var_filter)
	J = _filter_indices(data.obs, model.obs_filter, model.use_external_obs ? external_obs : nothing)

	var = model.var

	if I != Colon() && !table_cols_equal(data.var, model.var_match)
		I = _reordered_var_ind(I, data.var, model.var_match; verbose)
		var = :copy # keep is impossible
	end

	var == :copy && (var = data.var[I,:])

	obs = model.obs
	obs == :copy && (obs = data.obs[J,:])

	update_matrix(data, _subsetmatrix(data.matrix, I, J), model; var, obs)
end


"""
	filter_matrix(fvar, fobs, data::DataMatrix)

Return a new DataMatrix, containing only the variables and observations passing the filters.

`fvar`/`fobs` can be:
* An `AbstractVector` of indices to keep.
* A `AbstractVector` of booleans (true to keep, false to discard).
* `:` indicating that all variables/observations should be kept.
* Anything you can pass on to `DataFrames.filter` (see [DataFrames documentation](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter) for details).

Also note that indexing of a DataMatrix supports `AbstractVector`s of indices/booleans and `:`, and is otherwise identical to `filter_matrix`.

# Examples

Keep every 10th variable and 3rd observation:
```julia
julia> filter_matrix(1:10:size(data,1), 1:3:size(data,2), data)
```

Or, using indexing syntax:
```julia
julia> data[1:10:end, 1:3:end]
```

For more examples, see `filter_var` and `filter_obs`.

See also: [`filter_var`](@ref), [`filter_obs`](@ref), [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter)
"""
function filter_matrix(fvar, fobs, data::DataMatrix; external_var=nothing, external_obs=nothing)
	model = FilterModel(data,fvar,fobs; external_var, use_external_obs=external_obs!==nothing)
	project(data, model; external_obs, allow_obs_indexing=true)
end


"""
	filter_var(f, data::DataMatrix)

Return a new DataMatrix, containing only the variables passing the filter.

`f` can be:
* An `AbstractVector` of indices to keep.
* A `AbstractVector` of booleans (true to keep, false to discard).
* `:` indicating that all variables should be kept.
* Anything you can pass on to `DataFrames.filter` (see [DataFrames documentation](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter) for details).

# Examples

Keep every 10th variable:
```julia
julia> filter_var(1:10:size(data,1), data)
```

Keep only variables of the type "Gene Expression":
```julia
julia> filter_var("feature_type"=>isequal("Gene Expression"), data)
```

See also: [`filter_matrix`](@ref), [`filter_obs`](@ref), [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter)
"""
filter_var(f, data::DataMatrix) = filter_matrix(f, :, data)


"""
	filter_obs(f, data::DataMatrix)

Return a new DataMatrix, containing only the observations passing the filter.

`f` can be:
* An `AbstractVector` of indices to keep.
* A `AbstractVector` of booleans (true to keep, false to discard).
* `:` indicating that all observations should be kept.
* Anything you can pass on to `DataFrames.filter` (see DataFrames documentation for details).

# Examples

Keep every 10th observation:
```julia
julia> filter_obs(1:10:size(data,2), data)
```

Remove observations where "celltype" equals "other":
```julia
julia> filter_obs("celltype"=>!isequal("other"), data)
```

See also: [`filter_matrix`](@ref), [`filter_var`](@ref), [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter)
"""
filter_obs(f, data::DataMatrix) = filter_matrix(:, f, data)


# - show -
_show_filter(io, f::Colon) = print(io, ':')
_show_filter(io, f) = print(IOContext(io, :compact=>true, :limit=>true), f)

function Base.show(io::IO, ::MIME"text/plain", model::FilterModel)
	print(io, "FilterModel(")
	_show_filter(io, model.var_filter)
	print(io, ", ")
	_show_filter(io, model.obs_filter)
	print(io, ')')
end
