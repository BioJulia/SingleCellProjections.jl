struct LogTransformModel{T} <: ProjectionModel
	scale_factor::Float64
	var_match::DataFrame
	var::Symbol
	obs::Symbol
end
function LogTransformModel(::Type{T}, counts::DataMatrix;
                           var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
                           external_var = nothing,
                           scale_factor=10_000, var=:copy, obs=:copy) where T
	var_match = select(counts.var, 1)
	if var_filter !== nothing
		ind = external_var !== nothing ? _filter_indices(counts.var, var_filter, external_var) : _filter_indices(counts.var, var_filter)
		var_match = var_match[ind,:]
	end
	LogTransformModel{T}(scale_factor, var_match, var, obs)
end
LogTransformModel(counts::DataMatrix; kwargs...) = LogTransformModel(Float64, counts; kwargs...)

projection_isequal(m1::LogTransformModel{T1}, m2::LogTransformModel{T2}) where {T1,T2} =
	T1 === T2 && m1.scale_factor == m2.scale_factor && m1.var_match == m2.var_match

update_model(m::LogTransformModel{T}; scale_factor=m.scale_factor, var=m.var, obs=m.obs, kwargs...) where T =
	(LogTransformModel{T}(scale_factor, m.var_match, var, obs), kwargs)


function logtransform_impl(X, model::LogTransformModel{T}) where T
	P,N = size(X)
	s = max.(1, sum(X; dims=1))
	nf = model.scale_factor ./ s

	# log( 1 + c*f/s)
	nzval = nonzeros(X)
	nzval_out = zeros(T, nnz(X))

	for j in 1:N
		irange = nzrange(X,j)
		# nzval_out[irange] .= log2.(1 .+ (@view nzval[irange]) .* nf[j])
		nzval_out[irange] .= convert.(T, log2.(1 .+ (@view nzval[irange]) .* nf[j]))
	end

	A = SparseMatrixCSC(P, N, copy(X.colptr), copy(X.rowval), nzval_out)
	MatrixRef(:A=>A)
end

function project_impl(counts::DataMatrix, model::LogTransformModel; verbose=true)
	matrix = counts.matrix
	var = model.var
	if !table_cols_equal(counts.var, model.var_match)
		# variables are not identical, we need to: reorder, skip missing, get rid of extra
		var_ind = table_indexin(model.var_match, counts.var; cols=names(model.var_match))
		var_mask = var_ind.!==nothing
		var_ind2 = var_ind[var_mask]

		matrix = matrix[var_ind2,:]

		# reordering variable annotations - we have to ignore :keep
		var = counts.var[var_ind2,:]

		if verbose
			# - show info -
			n_missing = length(var_ind) - length(var_ind2)
			n_removed = size(counts.var,1) - length(var_ind2)
			n_removed>0 && @info "- Removed $n_removed variables that where not found in Model"
			n_missing>0 && @info "- Skipped $n_missing missing variables"
			issorted(var_ind2) || @info "- Reordered variables to match Model"
		end
	end

	matrix = logtransform_impl(matrix, model)
	update_matrix(counts, matrix, model; var, model.obs)
end

"""
	logtransform([T=Float64], counts::DataMatrix;
	             var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
	             external_var = nothing,
	             scale_factor=10_000,
	             var=:copy,
	             obs=:copy)

Log₂-transform `counts` using the formula:
```
  log₂(1 + cᵢⱼ*scale_factor/(∑ᵢcᵢⱼ))
```

Optionally, `T` can be specified to control the `eltype` of the sparse transformed matrix.
`T=Float32` can be used to lower the memory usage, with little impact on the results, since downstream analysis is still done with Float64.

* `var_filter` - Control which variables (features) to use for parameter estimation. Defaults to `:feature_type => isequal("Gene Expression")`, if a `feature_type` column is present in `counts.var`. Can be set to `nothing` to disable filtering. See [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter) for how to specify filters.
* `external_var` - If given, these annotations are used instead of `data.var` when applying `var_filter`. NB: The IDs of `external_var` must match IDs in `data.var`.
* `var` - Can be `:copy` (make a copy of source `var`) or `:keep` (share the source `var` object).
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

# Examples
```julia
julia> transformed = logtransform(counts)
```

Use eltype Float32 to lower memory usage:
```julia
julia> transformed = logtransform(Float32, counts)
```

See also: [`sctransform`](@ref)
"""
logtransform(::Type{T}, counts::DataMatrix; kwargs...) where T =
	project(counts, LogTransformModel(T, counts; kwargs...))
logtransform(counts::DataMatrix; kwargs...) = logtransform(Float64, counts; kwargs...)


struct TFIDFTransformModel{T} <: ProjectionModel
	scale_factor::Float64
	idf::Vector{Float64}
	var_match::DataFrame
	annotate::Bool # true: add idf as a var annotation
	var::Symbol
	obs::Symbol
end
function TFIDFTransformModel(::Type{T}, counts::DataMatrix;
                             var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
                             external_var = nothing,
                             scale_factor=10_000,
                             idf=vec(size(counts,2) ./ max.(1,sum(counts.matrix; dims=2))),
                             annotate=true,
                             var=:copy, obs=:copy) where T
	var_match = select(counts.var, 1)
	if var_filter !== nothing
		ind = external_var !== nothing ? _filter_indices(counts.var, var_filter, external_var) : _filter_indices(counts.var, var_filter)
		var_match = var_match[ind,:]
		idf = idf[ind]
	end
	TFIDFTransformModel{T}(scale_factor, idf, var_match, annotate, var, obs)
end
TFIDFTransformModel(counts::DataMatrix; kwargs...) =
	TFIDFTransformModel(Float64, counts; kwargs...)

projection_isequal(m1::TFIDFTransformModel{T1}, m2::TFIDFTransformModel{T2}) where {T1,T2} =
	T1 === T2 && m1.scale_factor == m2.scale_factor && m1.idf == m2.idf && m1.var_match == m2.var_match

update_model(m::TFIDFTransformModel{T}; scale_factor=m.scale_factor, idf=m.idf,
                                        annotate=m.annotate, var=m.var, obs=m.obs,
                                        kwargs...) where T =
	(TFIDFTransformModel{T}(scale_factor, idf, m.var_match, annotate, var, obs), kwargs)


function tf_idf_transform_impl(::Type{T}, X, scale_factor, idf) where T
	P,N = size(X)
	s = max.(1, sum(X; dims=1))
	nf = scale_factor ./ s

	# log( 1 + c*f/s * idf )
	R = rowvals(X)
	nzval = nonzeros(X)
	nzval_out = zeros(T, nnz(X))

	for j in 1:N
		for k in nzrange(X,j)
			i = R[k]
			# nzval_out[k] = log(1.0 + nzval[k]*nf[j]*idf[i])
			nzval_out[k] = convert(T, log(1.0 + nzval[k]*nf[j]*idf[i]))
		end
	end

	A = SparseMatrixCSC(P, N, copy(X.colptr), copy(X.rowval), nzval_out)
	MatrixRef(:A=>A)
end

function project_impl(counts::DataMatrix, model::TFIDFTransformModel{T}; verbose=true) where T
	# TODO: share this code with LogTransformModel - the only difference is idf
	matrix = counts.matrix
	var = model.var
	idf = model.idf

	if !table_cols_equal(counts.var, model.var_match)
		# variables are not identical, we need to: reorder, skip missing, get rid of extra
		var_ind = table_indexin(model.var_match, counts.var; cols=names(model.var_match))
		var_mask = var_ind.!==nothing
		var_ind2 = var_ind[var_mask]

		matrix = matrix[var_ind2,:]
		idf = idf[var_mask]

		# reordering variable annotations - we have to ignore :keep
		var = counts.var[var_ind2,:]

		if verbose
			# - show info -
			n_missing = length(var_ind) - length(var_ind2)
			n_removed = size(counts.var,1) - length(var_ind2)
			n_removed>0 && @info "- Removed $n_removed variables that where not found in Model"
			n_missing>0 && @info "- Skipped $n_missing missing variables"
			issorted(var_ind2) || @info "- Reordered variables to match Model"
		end
	end

	matrix = tf_idf_transform_impl(T, matrix, model.scale_factor, idf)

	if model.annotate
		if var isa Symbol
			var = copy(counts.var; copycols = var==:copy)
		end
		var.idf = idf
	end
	update_matrix(counts, matrix, model; var, model.obs)
end

"""
	tf_idf_transform([T=Float64], counts::DataMatrix;
	                 var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
	                 external_var,
	                 scale_factor = 10_000,
	                 idf = vec(size(counts,2) ./ max.(1,sum(counts.matrix; dims=2))),
	                 annotate = true,
	                 var = :copy,
	                 obs = :copy)

Compute the TF-IDF (term frequency-inverse document frequency) transform of `counts`, using
the formula `log( 1 + scale_factor * tf * idf )` where `tf` is the term frequency `counts.matrix ./ max.(1, sum(counts.matrix; dims=1))`.

Optionally, `T` can be specified to control the `eltype` of the sparse transformed matrix.
`T=Float32` can be used to lower the memory usage, with little impact on the results, since downstream analysis is still done with Float64.

* `var_filter` - Control which variables (features) to use for parameter estimation. Defaults to `:feature_type => isequal("Gene Expression")`, if a `feature_type` column is present in `counts.var`. Can be set to `nothing` to disable filtering. See [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter) for how to specify filters.
* `external_var` - If given, these annotations are used instead of `data.var` when applying `var_filter`. NB: The IDs of `external_var` must match IDs in `data.var`.
* `annotate` - If true, `idf` will be added as a `var` annotation.
* `var` - Can be `:copy` (make a copy of source `var`) or `:keep` (share the source `var` object).
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).
"""
tf_idf_transform(::Type{T}, counts::DataMatrix; kwargs...) where T =
	project(counts, TFIDFTransformModel(T, counts; kwargs...))
tf_idf_transform(counts::DataMatrix; kwargs...) =
	tf_idf_transform(Float64, counts; kwargs...)




scparams(counts::DataMatrix; kwargs...) = scparams(counts.matrix, counts.var; kwargs...)

struct SCTransformModel{T} <: ProjectionModel
	params::DataFrame
	var_id_col::String
	clip::Float64
	rtol::Float64
	atol::Float64
	annotate::Bool
	post_filter::FilterModel
end

"""
	SCTransformModel([T=Float64], counts::DataMatrix;
	                 var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
	                 rtol=1e-3, atol=0.0, annotate=true,
	                 post_var_filter=:, post_obs_filter=:,
	                 obs=:copy,
	                 kwargs...)

Computes the `SCTransform` parameter estimates for `counts` and creates a SCTransformModel that can be applied to the same or another data set.
Defaults to only using "Gene Expression" features.

Optionally, `T` can be specified to control the `eltype` of the sparse transformed matrix.
`T=Float32` can be used to lower the memory usage, with little impact on the results, since downstream analysis is still done with Float64.

* `var_filter` - Control which variables (features) to use for parameter estimation. Defaults to `:feature_type => isequal("Gene Expression")`, if a `feature_type` column is present in `counts.var`. Can be set to `nothing` to disable filtering. See [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter) for how to specify filters.
* `external_var` - If given, these annotations are used instead of `data.var` when applying `var_filter`. NB: The IDs of `external_var` must match IDs in `data.var`.
* `rtol` - Relative tolerance when constructing low rank approximation.
* `atol` - Absolute tolerance when constructing low rank approximation.
* `annotate` - Set to true to include SCTransform parameter estimates as feature annotations.
* `post_var_filter` - Equivalent to applying variable (feature) filtering after sctransform, but computationally more efficient.
* `post_obs_filter` - Equivalent to applying observation (cell) filtering after sctransform, but computationally more efficient.
* `external_post_var` - If given, these annotations are used instead of `data.var` when applying `post_var_filter`. NB: The IDs of `external_post_var` must match IDs in `data.var`.
* `external_post_obs` - If given, these annotations are used instead of `data.var` when applying `post_obs_filter`. NB: The IDs of `external_post_obs` must match IDs in `data.obs`.
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).
* `kwargs...` - Additional `kwargs` are passed on to [`SCTransform.scparams`](https://github.com/rasmushenningsson/SCTransform.jl).

# Examples
Setup `SCTransformModel` (Gene Expression features):
```
julia> SCTransformModel(counts)
```

Setup `SCTransformModel` (Antibody Capture features):
```
julia> SCTransformModel(counts; var_filter = :feature_type => isequal("Antibody Capture"))
```

See also: [`sctransform`](@ref), [`SCTransform.scparams`](https://github.com/rasmushenningsson/SCTransform.jl), [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter)
"""
function SCTransformModel(::Type{T}, counts::DataMatrix;
                          var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
                          external_var = nothing,
                          rtol=1e-3, atol=0.0, annotate=true,
                          post_var_filter=:, post_obs_filter=:,
                          external_post_var = nothing,
                          external_post_obs = nothing,
                          obs=:copy,
                          kwargs...) where T
	nvar = size(counts,1)

	if var_filter === nothing
		feature_mask = trues(nvar)
	else
		ind = external_var !== nothing ? _filter_indices(counts.var, var_filter, external_var) : _filter_indices(counts.var, var_filter)
		feature_mask = falses(nvar)
		feature_mask[ind] .= true
	end

	params = scparams(counts; feature_mask, kwargs...)
	clip = sqrt(size(counts,2)/30)
	post_filter = FilterModel(params, post_var_filter, post_obs_filter; var=:copy, obs, external_var=external_post_var, use_external_obs=external_post_obs!==nothing)
	SCTransformModel{T}(params, only(names(counts.var,1)), clip, rtol, atol, annotate, post_filter)
end
SCTransformModel(counts::DataMatrix; kwargs...) =
	SCTransformModel(Float64, counts; kwargs...)

function projection_isequal(m1::SCTransformModel{T1}, m2::SCTransformModel{T2}) where {T1,T2}
	T1 === T2 &&
	m1.params == m2.params &&
	m1.var_id_col == m2.var_id_col &&
	m1.clip == m2.clip &&
	m1.rtol == m2.rtol &&
	m1.atol == m2.atol &&
	projection_isequal(m1.post_filter, m2.post_filter)
end


function update_model(m::SCTransformModel{T};
                      clip=m.clip, rtol=m.rtol, atol=m.atol,
                      annotate=m.annotate,
                      post_var_filter=m.post_filter.var_filter,
                      post_obs_filter=nothing,
                      obs=m.post_filter.obs,
                      kwargs...
                     ) where T
	post_var_filter = _filter_indices(m.params, post_var_filter)

	allow_obs_indexing = post_obs_filter !== nothing
	post_obs_filter === nothing && (post_obs_filter = m.post_filter.obs_filter)

	post_filter = FilterModel(post_var_filter, post_obs_filter, m.post_filter.var_match, m.post_filter.use_external_obs, m.post_filter.var, obs) # TODO: support external_var here?
	model = SCTransformModel{T}(m.params, m.var_id_col, clip, rtol, atol, annotate, post_filter)
	(model, (;allow_obs_indexing, kwargs...))
end


function project_impl(counts::DataMatrix, model::SCTransformModel{T}; external_post_obs=nothing, allow_obs_indexing=false, verbose=true) where T
	# use post_filter to figure out variable and observations subsetting
	_validate(model.params, model.post_filter, allow_obs_indexing, SCTransformModel, "post_obs_filter")

	I = _filter_indices(model.params, model.post_filter.var_filter)
	J = model.post_filter.use_external_obs ?  _filter_indices(counts.obs, model.post_filter.obs_filter, external_post_obs) : _filter_indices(counts.obs, model.post_filter.obs_filter)
	params = model.params[I,:]

	# Remove any variables not found in counts
	only(names(counts.var,1)) == model.var_id_col || error("Expected variable id column $(model.var_id_col), got $(only(names(counts.var,1)))")
	var_ind = table_indexin(params, counts.var; cols=model.var_id_col)
	var_mask = var_ind.!==nothing
	var_ind2 = var_ind[var_mask]

	n_removed = size(counts.var,1) - length(var_ind2)
	verbose && n_removed>0 && @info "- Removed $n_removed variables that were not found in Model"

	n_missing = length(var_ind) - length(var_ind2)
	if n_missing>0
		params = params[var_mask, :]

		if verbose
			# - show info -
			@info "- Skipped $n_missing missing variables"
			issorted(var_ind2) || @info "- Reordered variables to match Model"
		end
	end

	X,var = sctransformsparse(T, counts.matrix, counts.var, params;
	                          feature_id_columns=[model.var_id_col],
	                          cell_ind=J,
	                          model.clip, model.rtol, model.atol)

	model.annotate && leftjoin!(var, params; on=intersect(names(var), names(params)))

	obs = counts.obs
	@assert model.post_filter.obs in (:copy,:keep)
	if J != Colon() || model.post_filter.obs == :copy
		obs = obs[J,:]
	end
	update_matrix(counts, X, model; var, obs)
end


"""
	sctransform([T=Float64], counts::DataMatrix; verbose=true, kwargs...)

Compute the SCTransform of the DataMatrix `counts`.
The result is stored as a Matrix Expression with the sum of a sparse and a low-rank term.
I.e. no large dense matrix is created.

Optionally, `T` can be specified to control the `eltype` of the sparse transformed matrix.
`T=Float32` can be used to lower the memory usage, with little impact on the results, since downstream analysis is still done with Float64.

See `SCTransformModel` for description of `kwargs...`.

# Examples
Compute SCTransform (Gene Expression features):
```
julia> sctransform(counts)
```

Compute SCTransform (Antibody Capture features):
```
julia> sctransform(counts; var_filter = :feature_type => isequal("Antibody Capture"))
```

Compute SCTransform (Gene Expression features), using eltype Float32 to lower memory usage:
```
julia> sctransform(Float32, counts)
```

See also: [`SCTransformModel`](@ref), [`SCTransform.scparams`](https://github.com/rasmushenningsson/SCTransform.jl)
"""
sctransform(::Type{T}, counts::DataMatrix; verbose=true, kwargs...) where T =
	project_impl(counts, SCTransformModel(T, counts; verbose, kwargs...); verbose, allow_obs_indexing=true)
sctransform(counts::DataMatrix; kwargs...) = sctransform(Float64, counts; kwargs...)

# - show -
function Base.show(io::IO, ::MIME"text/plain", model::LogTransformModel{T}) where T
	print(io, "LogTransformModel")
	T !== Float64 && print(io, '{', T, '}')
	print(io, "(scale_factor=", round(model.scale_factor;digits=2), ')')
end
function Base.show(io::IO, ::MIME"text/plain", model::TFIDFTransformModel{T}) where T
	print(io, "TFIDFTransformModel")
	T !== Float64 && print(io, '{', T, '}')
	print(io, "(scale_factor=", round(model.scale_factor;digits=2), ')')
end
function Base.show(io::IO, ::MIME"text/plain", model::SCTransformModel{T}) where T
	print(io, "SCTransformModel")
	T !== Float64 && print(io, '{', T, '}')
	print(io, "(nvar=", size(model.params,1), ", clip=", round(model.clip;digits=2), ')')
end
