_subsetmatrix(X::AbstractMatrix, I::Index, J::Index) = X[I,J]


_filter_indices(::DataFrame, I::Index) = I
_filter_indices(df::DataFrame, f) = first(parentindices(filter(f,df;view=true)))

struct FilterModel{Tv<:Index,Ts} <: ProjectionModel
	var_filter::Tv
	obs_filter::Ts
	var_match::DataFrame
	var::Symbol
	obs::Symbol

    function FilterModel(var_filter::Tv, obs_filter::Ts, var_match, var, obs) where {Tv<:Index,Ts}
		# :keep only possible when indexing with :
		var_filter != Colon() && var == :keep && throw(ArgumentError("var = :keep is only allowed when indexing with :"))
		obs_filter != Colon() && obs == :keep && throw(ArgumentError("obs = :keep is only allowed when indexing with :"))
        new{Tv,Ts}(var_filter, obs_filter, var_match, var, obs)
    end
end
FilterModel(var_annots::Tv, var_id_cols, var_filter, obs_filter; var=:copy, obs=:copy) where Tv=
	FilterModel(_filter_indices(var_annots, var_filter), obs_filter, select(var_annots, var_id_cols), var, obs)
FilterModel(data::DataMatrix, args...; kwargs...) =
	FilterModel(data.var, data.var_id_cols, args...; kwargs...)

function projection_isequal(m1::FilterModel, m2::FilterModel)
	m1.var_filter == m2.var_filter && m1.obs_filter == m2.obs_filter &&
	m1.var_match == m2.var_match
end


function update_model(m::FilterModel; var_filter=m.var_filter, obs_filter=nothing,
                      var=m.var, obs=m.obs, kwargs...)
	allow_obs_indexing = obs_filter !== nothing
	obs_filter === nothing && (obs_filter = m.obs_filter)
	model = FilterModel(var_filter, obs_filter, m.var_match, var, obs)
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

function project_impl(data::DataMatrix, model::FilterModel; allow_obs_indexing=false, verbose=true)
	_validate(data.var, model, allow_obs_indexing)

	I = _filter_indices(data.var, model.var_filter)
	J = _filter_indices(data.obs, model.obs_filter)

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


filter_matrix(data::DataMatrix, fvar, fobs) = project(data, FilterModel(data,fvar,fobs); allow_obs_indexing=true)
filter_var(f, data::DataMatrix) = filter_matrix(data, f, :)
filter_obs(f, data::DataMatrix) = filter_matrix(data, :, f)

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
