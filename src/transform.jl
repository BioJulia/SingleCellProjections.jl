struct LogTransformModel <: ProjectionModel
	scale_factor::Float64
	var_match::DataFrame
	var::Symbol
	obs::Symbol
end
LogTransformModel(counts::DataMatrix; scale_factor=10_000, var=:copy, obs=:copy) =
	LogTransformModel(scale_factor, select(counts.var,counts.var_id_cols), var, obs)

projection_isequal(m1::LogTransformModel, m2::LogTransformModel) =
	m1.scale_factor == m2.scale_factor && m1.var_match == m2.var_match

update_model(m::LogTransformModel; scale_factor=m.scale_factor, var=m.var, obs=m.obs, kwargs...) =
	(LogTransformModel(scale_factor, m.var_match, var, obs), kwargs)


function logtransform_impl(X, model::LogTransformModel)
	P,N = size(X)
	s = max.(1, sum(X; dims=1))
	nf = model.scale_factor ./ s

	# log( 1 + c*f/s)
	nzval = nonzeros(X)
	nzval_out = zeros(nnz(X))

	for j in 1:N
		irange = nzrange(X,j)
		nzval_out[irange] .= log2.(1 .+ (@view nzval[irange]) .* nf[j])
	end

	ThreadedSparseMatrixCSC(SparseMatrixCSC(P, N, copy(X.colptr), copy(X.rowval), nzval_out))
end

function project_impl(counts::DataMatrix, model::LogTransformModel; verbose=true)
	matrix = counts.matrix
	var = model.var
	if !table_cols_equal(counts.var, model.var_match)
		# variables are not identical, we need to: reorder, skip missing, get rid of extra
		var_ind = table_indexin(model.var_match, counts.var; cols=names(model.var_match))
		var_ind2 = Int[i for i in var_ind if i!==nothing] # collect(skipnothing(var_ind))

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
	logtransform(counts::DataMatrix; scale_factor=10_000, var=:copy, obs=:copy)

Log-transform `counts` using the formula:
```
  log(1 + cᵢⱼ*scale_factor/(∑ᵢcᵢⱼ))
```

* `var` - Can be `:copy` (make a copy of source `var`) or `:keep` (share the source `var` object).
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

# Examples
```julia
julia> transformed = logtransform(counts)
```

See also: [`sctransform`](@ref)
"""
logtransform(counts::DataMatrix; kwargs...) = project(counts, LogTransformModel(counts; kwargs...))



struct TFIDFTransformModel <: ProjectionModel
	scale_factor::Float64
	idf::Vector{Float64}
	var_match::DataFrame
	annotate::Bool # true: add idf as a var annotation
	var::Symbol
	obs::Symbol
end
function TFIDFTransformModel(counts::DataMatrix;
                             scale_factor=10_000,
                             idf=vec(size(counts,2) ./ max.(1,sum(counts.matrix; dims=2))),
                             annotate=true,
                             var=:copy, obs=:copy)
	var_match = select(counts.var,counts.var_id_cols)
	TFIDFTransformModel(scale_factor, idf, var_match, annotate, var, obs)
end

projection_isequal(m1::TFIDFTransformModel, m2::TFIDFTransformModel) =
	m1.scale_factor == m2.scale_factor && m1.idf == m2.idf && m1.var_match == m2.var_match

update_model(m::TFIDFTransformModel; scale_factor=m.scale_factor, idf=m.idf,
                                     annotate=m.annotate, var=m.var, obs=m.obs, kwargs...) =
	(TFIDFTransformModel(scale_factor, idf, m.var_match, annotate, var, obs), kwargs)


function tf_idf_transform_impl(X, scale_factor, idf)
	P,N = size(X)
	s = max.(1, sum(X; dims=1))
	nf = scale_factor ./ s

	# log( 1 + c*f/s * idf )
	R = rowvals(X)
	nzval = nonzeros(X)
	nzval_out = zeros(nnz(X))

	for j in 1:N
		for k in nzrange(X,j)
			i = R[k]
			nzval_out[k] = log(1.0 + nzval[k]*nf[j]*idf[i])
		end
	end

	ThreadedSparseMatrixCSC(SparseMatrixCSC(P, N, copy(X.colptr), copy(X.rowval), nzval_out))
end

function project_impl(counts::DataMatrix, model::TFIDFTransformModel; verbose=true)
	# TODO: share this code with LogTransformModel - the only difference is idf
	matrix = counts.matrix
	var = model.var
	idf = model.idf

	if !table_cols_equal(counts.var, model.var_match)
		# variables are not identical, we need to: reorder, skip missing, get rid of extra
		var_ind = table_indexin(model.var_match, counts.var; cols=names(model.var_match))
		var_ind2 = Int[i for i in var_ind if i!==nothing] # collect(skipnothing(var_ind))

		matrix = matrix[var_ind2,:]
		idf = idf[var_ind2,:]

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

	matrix = tf_idf_transform_impl(matrix, model.scale_factor, idf)

	if model.annotate
		if var isa Symbol
			var = copy(counts.var; copycols = var==:copy)
		end
		var.idf = idf
	end
	update_matrix(counts, matrix, model; var, model.obs)
end

"""
	tf_idf_transform(counts::DataMatrix;
                     scale_factor = 10_000,
                     idf = vec(size(counts,2) ./ max.(1,sum(counts.matrix; dims=2))),
                     annotate = true,
                     var = :copy,
                     obs = :copy)

Compute the TF-IDF (term frequency-inverse document frequency) transform of `counts`, using
the formula `log( 1 + scale_factor * tf * idf )` where `tf` is the term frequency `counts.matrix ./ max.(1, sum(counts.matrix; dims=1))`.

If `annotate` is true, `idf` will be added as a `var` annotation.
"""
tf_idf_transform(counts::DataMatrix; kwargs...) = project(counts, TFIDFTransformModel(counts; kwargs...))




scparams(counts::DataMatrix; kwargs...) = scparams(counts.matrix, counts.var; kwargs...)

struct SCTransformModel <: ProjectionModel
	params::DataFrame
	var_id_cols::Vector{String}
	clip::Float64
	rtol::Float64
	atol::Float64
	annotate::Bool
	post_filter::FilterModel
end

"""
	SCTransformModel(counts::DataMatrix;
	                 var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
	                 rtol=1e-3, atol=0.0, annotate=true,
	                 post_var_filter=:, post_obs_filter=:,
	                 obs=:copy,
	                 kwargs...)

Computes the `SCTransform` parameter estimates for `counts` and creates a SCTransformModel that can be applied to the same or another data set.
Defaults to only using "Gene Expression" features.

* `var_filter` - Control which variables (features) to use for parameter estimation. Defaults to `:feature_type => isequal("Gene Expression")`, if a `feature_type` column is present in `counts.var`. Can be set to `nothing` to disable filtering. See [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter) for how to specify filters.
* `rtol` - Relative tolerance when constructing low rank approximation.
* `atol` - Absolute tolerance when constructing low rank approximation.
* `annotate` - Set to true to include SCTransform parameter estimates as feature annotations.
* `post_var_filter` - Equivalent to applying variable (feature) filtering after sctransform, but computationally more efficient.
* `post_obs_filter` - Equivalent to applying observation (cell) filtering after sctransform, but computationally more efficient.
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).
* `kwargs...` - Additional `kwargs` are passed on to `SCTransform.scparams`.

# Examples
Setup `SCTransformModel` (Gene Expression features):
```
julia> SCTransformModel(counts)
```

Setup `SCTransformModel` (Antibody Capture features):
```
julia> SCTransformModel(counts; var_filter = :feature_type => isequal("Antibody Capture"))
```

See also: [`sctransform`](@ref), [`SCTransform.scparams`](@ref), [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter)
"""
function SCTransformModel(counts::DataMatrix;
                          var_filter = hasproperty(counts.var, :feature_type) ? :feature_type => isequal("Gene Expression") : nothing,
                          rtol=1e-3, atol=0.0, annotate=true,
                          post_var_filter=:, post_obs_filter=:,
                          obs=:copy,
                          kwargs...)
	nvar = size(counts,1)

	if var_filter === nothing
		feature_mask = trues(nvar)
	else
		sub = filter(var_filter, counts.var; view=true)
		ind = first(parentindices(sub))
		feature_mask = falses(nvar)
		feature_mask[ind] .= true
	end

	params = scparams(counts; feature_mask, kwargs...)
	clip = sqrt(size(counts,2)/30)
	post_filter = FilterModel(params, counts.var_id_cols, post_var_filter, post_obs_filter; var=:copy, obs)
	SCTransformModel(params, counts.var_id_cols, clip, rtol, atol, annotate, post_filter)
end

function projection_isequal(m1::SCTransformModel, m2::SCTransformModel)
	m1.params == m2.params &&
	m1.var_id_cols == m2.var_id_cols &&
	m1.clip == m2.clip &&
	m1.rtol == m2.rtol &&
	m1.atol == m2.atol &&
	projection_isequal(m1.post_filter, m2.post_filter)
end


function update_model(m::SCTransformModel;
                      clip=m.clip, rtol=m.rtol, atol=m.atol,
                      annotate=m.annotate,
                      post_var_filter=m.post_filter.var_filter,
                      post_obs_filter=nothing,
                      obs=m.post_filter.obs,
                      kwargs...
                     )
	post_var_filter = _filter_indices(m.params, post_var_filter)

	allow_obs_indexing = post_obs_filter !== nothing
	post_obs_filter === nothing && (post_obs_filter = m.post_filter.obs_filter)

	post_filter = FilterModel(post_var_filter, post_obs_filter, m.post_filter.var_match, m.post_filter.var, obs)
	model = SCTransformModel(m.params, m.var_id_cols, clip, rtol, atol, annotate, post_filter)
	(model, (;allow_obs_indexing, kwargs...))
end


function project_impl(counts::DataMatrix, model::SCTransformModel; allow_obs_indexing=false, verbose=true)
	# use post_filter to figure out variable and observations subsetting
	_validate(model.params, model.post_filter, allow_obs_indexing, SCTransformModel, "post_obs_filter")

	I = _filter_indices(model.params, model.post_filter.var_filter)
	J = _filter_indices(counts.obs, model.post_filter.obs_filter)
	params = model.params[I,:]

	# Remove any variables not found in counts
	var_ind = table_indexin(params, counts.var; cols=model.var_id_cols)
	var_mask = var_ind.!==nothing
	var_ind2 = var_ind[var_mask]

	n_removed = size(counts.var,1) - length(var_ind2)
	verbose && n_removed>0 && @info "- Removed $n_removed variables that where not found in Model"

	n_missing = length(var_ind) - length(var_ind2)
	if n_missing>0
		params = params[var_mask, :]

		if verbose
			# - show info -
			@info "- Skipped $n_missing missing variables"
			issorted(var_ind2) || @info "- Reordered variables to match Model"
		end
	end

	X,var = sctransformsparse(counts.matrix, counts.var, params;
	                          feature_id_columns=model.var_id_cols,
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
	sctransform(counts::DataMatrix; verbose=true, kwargs...)

Compute the SCTransform of the DataMatrix `counts`.
The result is stored as a Matrix Expression with the sum of a sparse and a low-rank term.
I.e. no large dense matrix is created.

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

See also: [`SCTransformModel`](@ref), [`SCTransform.scparams`](@ref)
"""
sctransform(counts::DataMatrix; verbose=true, kwargs...) =
	project_impl(counts, SCTransformModel(counts; verbose, kwargs...); verbose, allow_obs_indexing=true)


# - show -
function Base.show(io::IO, ::MIME"text/plain", model::LogTransformModel)
	print(io, "LogTransformModel(scale_factor=", round(model.scale_factor;digits=2), ')')
end
function Base.show(io::IO, ::MIME"text/plain", model::TFIDFTransformModel)
	print(io, "TFIDFTransformModel(scale_factor=", round(model.scale_factor;digits=2), ')')
end
function Base.show(io::IO, ::MIME"text/plain", model::SCTransformModel)
	print(io, "SCTransformModel(nvar=", size(model.params,1), ", clip=", round(model.clip;digits=2), ')')
end
