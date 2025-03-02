_subsetmatrix(X::AbstractMatrix, I::Index, J::Index) = X[I,J]



function _join_external_columns!(df, col::Union{Symbol,String}, external)
	a = find_annotation(col, external)
	a === nothing && throw(ArgumentError("External annotation \"$col\" missing."))
	leftjoin!(df, a; on=names(df,1))
	nothing
end

function _join_external_columns!(df, cols::AbstractVector, external)
	for col in cols
		_join_external_columns!(df, col, external)
	end
end


function _filter_indices_external(df::DataFrame, f::Pair{<:Union{Symbol,String,<:AbstractVector},<:Any}, external)
	df = select(df, 1; copycols=false) # just keep the ID column

	# 1. Find names of columns needed for predicate
	# 2. Extract columns needed for predicate (and join to IDs)
	_join_external_columns!(df, first(f), external)

	# 3. Use standard _filter_indices
	_filter_indices(df, f)
end





_filter_indices(::DataFrame, I::Index) = I
_filter_indices(df::DataFrame, f) = first(parentindices(filter(f,df;view=true)))


_join_columns!(df, x::DataFrame) = leftjoin!(df, x; on=names(df,1))
_join_columns!(df, x::Annotations) = _join_columns!(df, get_table(x))

function _filter_indices(df::DataFrame, f::Pair{<:Union{<:AbstractDataFrame, Annotations},<:Any})
	df = select(df, 1; copycols=false) # just keep the ID column

	# leftjoin! columns
	_join_columns!(df, first(f))

	# 3. Use standard _filter_indices
	f2 = names(df,2:size(df,2)) => last(f)
	_filter_indices(df, f2)
end



# For filters using Annotations/DataFrames, this extracts the names and indicates that we need to use an external source for the annotations during projection
# We also want to catch obs_filter with Index here - i.e. that cannot be externalized, so a mask must be provided using obs_filter kwargs when projecting
function _externalize_filter(f)
	if !(f isa Colon) && f isa Index
		InvalidIndex(),true
	else
		f,false
	end
end
_externalize_filter((cols,f)::Pair{<:AbstractDataFrame,<:Any}) = names(cols,2:size(cols,2))=>f, true
_externalize_filter((cols,f)::Pair{Annotations,<:Any}) = _externalize_filter(get_table(cols)=>f)



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
function FilterModel(var_annots::Tv, var_filter, obs_filter; var=:copy, obs=:copy) where Tv
	var_filter = _filter_indices(var_annots, var_filter)
	obs_filter, use_external_obs = _externalize_filter(obs_filter)
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
	if obs_filter !== nothing
		obs_filter_externalized, use_external_obs = _externalize_filter(obs_filter)
		model = FilterModel(var_filter, obs_filter_externalized, m.var_match, use_external_obs, var, obs)
		(model, (;original_obs_filter=obs_filter, kwargs...))
	else
		model = FilterModel(var_filter, m.obs_filter, m.var_match, m.use_external_obs, var, obs)
		(model, kwargs)
	end
end


function _validate(var, model::FilterModel, original_obs_filter, ::Type{T}=typeof(model), name="obs_filter") where T
	if original_obs_filter === nothing && model.obs_filter == InvalidIndex()
		throw(ArgumentError("$(nameof(T)) was constructed with explicit observation indices. Use a different $(nameof(T)) or call project() with $name set."))
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

function project_impl(data::DataMatrix, model::FilterModel; external_obs=nothing, original_obs_filter=nothing, verbose=true, kwargs...)
	_validate(data.var, model, original_obs_filter)

	I = _filter_indices(data.var, model.var_filter)

	if original_obs_filter !== nothing
		J = _filter_indices(data.obs, original_obs_filter)
	elseif model.use_external_obs
		J = _filter_indices_external(data.obs, model.obs_filter, external_obs)
	else
		J = _filter_indices(data.obs, model.obs_filter)
	end


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
function filter_matrix(fvar, fobs, data::DataMatrix)
	model = FilterModel(data, fvar, fobs)
	project(data, model; original_obs_filter=fobs)
end


"""
	filter_var(f, data::DataMatrix; kwargs...)

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
filter_var(f, data::DataMatrix; kwargs...) = filter_matrix(f, :, data; kwargs...)


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
filter_obs(f, data::DataMatrix; kwargs...) = filter_matrix(:, f, data; kwargs...)


# - show -
_show_filter(io, f::Colon) = print(io, ':')
function _show_filter(io, f, use_external=false)
	use_external && print(io, "external(")
	print(IOContext(io, :compact=>true, :limit=>true), f)
	use_external && print(io, ')')
end

function Base.show(io::IO, ::MIME"text/plain", model::FilterModel)
	print(io, "FilterModel(")
	_show_filter(io, model.var_filter)
	print(io, ", ")
	_show_filter(io, model.obs_filter, model.use_external_obs)
	print(io, ')')
end
