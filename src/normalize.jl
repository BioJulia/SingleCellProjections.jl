function variable_sum_squares(matrix)
	X = matrixexpression(matrix)
	d = DiagGram(X')
	r = compute(d)
	max.(0.0, r)
end
variable_sum_squares(data::DataMatrix) = variable_sum_squares(data.matrix)

"""
	variable_var(data::DataMatrix)

Computes the variance of each variable in `data`.

!!! note
	`data` must be mean-centered. E.g. by using `normalize_matrix` before calling `variable_var`.

See also: [`variable_std`](@ref), [`normalize_matrix`](@ref)
"""
variable_var(data::DataMatrix) = variable_sum_squares(data) ./= size(data,2)-1

"""
	variable_std(data::DataMatrix)

Computes the variance of each variable in `data`.

!!! note
	`data` must be mean-centered. E.g. by using `normalize_matrix` before calling `variable_std`.

See also: [`variable_var`](@ref), [`normalize_matrix`](@ref)
"""
variable_std(data::DataMatrix) = sqrt.(variable_var(data))



# TEMP simplified normalization for testing projections via Specs
struct CenteringModel <: ProjectionModel
	negβT::Matrix{Float64}
	var_ids::Vector{String}
end
function CenteringModel(data::DataMatrix)
	N = size(data,2)
	X = ones(N,1)
	negβT = (data.matrix*X) .* (-1/N)
	CenteringModel(negβT, data.var[:,1])
end


function project_impl(data::DataMatrix, model::CenteringModel; kwargs...)
	@assert data.var[:,1] == model.var_ids # SIMPLICIFACTION SINCE THIS CODE WILL NOT ACTUALLY BE USED
	N = size(data,2)
	X = ones(N,1)
	matrix = matrixsum(_named_matrix(data.matrix,:A), matrixproduct(Symbol("(-β)")=>model.negβT, :X=>X'))
	update_matrix(data, matrix, model; var=:keep, obs=:keep)
end

center_matrix(data::DataMatrix; kwargs...) =
	project(data, CenteringModel(data); kwargs...)




# Testing new NormalizationModel
struct NormalizationModel2 <: ProjectionModel
	negβT::DataMatrix{Matrix{Float64},DataFrame,DataFrame}
	rank::Int # Just for `show`
end

function NormalizationModel2(data::DataMatrix, design::DataMatrix; rtol=sqrt(eps()))
	@assert data.obs[:,1] == design.var[:,1]

	A = data.matrix
	X = design.matrix

	# TODO: No need to run svd etc. if there just is an intercept.
	F = svd(X)
	negΣinv = Diagonal([σ>rtol ? -1.0/σ : 0.0 for σ in F.S]) # cutoff for numerical stability
	rank = count(!iszero, negΣinv)
	AU = A*F.U
	negβT = (AU*negΣinv)*F.Vt

	NormalizationModel2(DataMatrix(negβT, data.var, design.obs), rank)
end


function project2(data::DataMatrix, model::NormalizationModel2; design::DataMatrix)
	# TODO: Reorder variables if needed
	# TODO: Handle missing variables in design matrix (we can reconstruct them if centering is enabled)

	@assert data.var[:,1] == model.negβT.var[:,1]
	@assert data.obs[:,1] == design.var[:,1]
	@assert design.obs[:,1] == model.negβT.obs[:,1]

	matrix = matrixsum(_named_matrix(data.matrix,:A), matrixproduct(Symbol("(-β)")=>model.negβT.matrix, :X=>design.matrix'))
end




struct NormalizationModel <: ProjectionModel
	negβT::Matrix{Float64}
	covariates::Vector{AbstractCovariate}
	rank::Int # Just for show
	var_match::DataFrame
	scaling::Vector{Float64} # empty vector means no scaling
	annotate::Bool # true: add scaling as a var annotation
	var::Symbol
	obs::Symbol
end

function projection_isequal(m1::NormalizationModel, m2::NormalizationModel)
	m1.negβT == m2.negβT && m1.covariates == m2.covariates && m1.rank == m2.rank &&
	m1.var_match == m2.var_match && m1.scaling == m2.scaling
end


function update_model(m::NormalizationModel; scaling=m.scaling, annotate=m.annotate, var=m.var, obs=m.obs, kwargs...)
	if scaling===false
		scaling = Float64[]
	elseif scaling === true
		@assert !isempty(m.scaling)
		scaling = m.scaling
	end

	model = NormalizationModel(m.negβT, m.covariates, m.rank, m.var_match,
	                           scaling, annotate, var, obs)
	model,kwargs
end


function _setscaling!(model::NormalizationModel, data::DataMatrix, design::DesignMatrix, scale::Bool, min_std)
	if scale
		any(==(InterceptCovariate()), design.covariates) || throw(ArgumentError("Scaling to unit std requires center=true.")) # Relax this? We can e.g. check that ones(N) is an singular vector of the normalized matrix with singular value ≈ 0.
		# We need to compute the std of the matrix _after_ normalization, i.e. project
		normalized = project(data, model, design)
		std = variable_std(normalized)
		resize!(model.scaling, size(data,1))
		model.scaling .= 1.0 ./ max.(std, min_std)
	else
		empty!(model.scaling)
	end
	model
end
function _setscaling!(model::NormalizationModel, data::DataMatrix, ::DesignMatrix, scale::AbstractVector, ::Any)
	resize!(model.scaling, size(data,1))
	model.scaling .= scale
	model
end


"""
	NormalizationModel(data::DataMatrix, design::DesignMatrix;
	                   scale=false, min_std=1e-6, annotate=true,
	                   rtol=sqrt(eps()), var=:copy, obs=:copy)

Create a NormalizationModel based on `data` and a `design` matrix.

* `scale` - Set to true to normalize variables to unit standard deviation. Can also be set to a vector with a scaling factor for each variable.
* `min_std` - If `scale==true`, the `scale` vector is set to `1.0 ./ max.(std, min_std)`. That is, `min_std` is used to suppress variables that are very small (and any fluctuations can be assumed to be noise).
* `annotate` - Only used if `scale!=false`. With `annotate=true`, the `scale` vector is added as a var annotation.
* `rtol` - Singular values of the design matrix that are `≤rtol` are discarded. Needed for numerical stability.
* `var` - Can be `:copy` (make a copy of source `var`) or `:keep` (share the source `var` object).
* `obs` - Can be `:copy` (make a copy of source `obs`) or `:keep` (share the source `obs` object).

See also: [`normalize_matrix`](@ref), [`designmatrix`](@ref)
"""
function NormalizationModel(data::DataMatrix, design::DesignMatrix;
                            scale=false, # can also be a vector
                            min_std=1e-6,
                            annotate=true,
                            rtol=sqrt(eps()),
                            var=:copy, obs=:copy,
                            )
	A = data.matrix
	X = design.matrix

	# TODO: No need to run svd etc. if there just is an intercept.
	F = svd(X)
	negΣinv = Diagonal([σ>rtol ? -1.0/σ : 0.0 for σ in F.S]) # cutoff for numerical stability
	rank = count(!iszero, negΣinv)
	AU = A*F.U
	negβT = (AU*negΣinv)*F.Vt

	model = NormalizationModel(negβT, design.covariates, rank, select(data.var,1), [], annotate, var, obs)
	_setscaling!(model, data, design, scale, min_std)
end


_reorder_matrix(A::MatrixSum, E) = MatrixSum([matrixproduct(E,term) for term in A.terms]) # distribute over sum
_reorder_matrix(matrix, E) = matrixproduct(E, matrix)
function _reorder_matrix(matrix, var_ind, P)
	E = index2matrix(var_ind, P)
	_reorder_matrix(matrix, MatrixRef(:E=>E))
end

_named_matrix(A::MatrixExpression, ::Symbol) = A
_named_matrix(A, name::Symbol) = MatrixRef(name=>A)





function _get_value_vector(data::DataMatrix, cov::AbstractCovariate, external_obs)
	cov isa InterceptCovariate && return nothing
	cov.external == false && return data.obs[!,cov.name]

	a = find_annotation(cov.name, external_obs)
	a === nothing && throw(ArgumentError("External annotation \"$(cov.name)\" missing, please provide external_obs when projecting."))

	obs = select(data.obs,1; copycols=false)
	leftjoin!(obs, a; on=names(obs,1))
	obs[!,2]
end



function project_impl(data::DataMatrix, model::NormalizationModel, design::DesignMatrix; verbose=true, kwargs...)
	@assert model.var in (:keep, :copy)
	@assert names(data.var,1) == names(model.var_match)

	@assert table_cols_equal(data.obs, design.obs_match) "Normalization expects design matrix and data matrix observations to be identical."

	matrix = data.matrix
	negβT = model.negβT

	# TODO: can we simplify this code?
	if !table_cols_equal(data.var, model.var_match)
		# variables are not identical, we need to: reorder, handling missing, get rid of extra
		var_ind = table_indexin(data.var, model.var_match)
		matrix = _reorder_matrix(_named_matrix(matrix,:A), var_ind, size(model.var_match,1))

		# zero out rows for missing variables in negβT
		mask = falses(size(model.var_match,1))
		var_ind2 = var_ind[var_ind.!==nothing]
		mask[var_ind2] .= true
		negβT = negβT .* mask

		center = any(==(InterceptCovariate()), model.covariates)
		@assert center || all(mask) "Missing variables can only be reconstructed if center=true in NormalizationModel."

		# reordering variable annotations - we have to ignore :keep
		var = copy(model.var_match)
		leftjoin!(var, data.var; on=names(model.var_match))

		if verbose
			# - show info -
			n_missing = size(model.var_match,1) - length(var_ind2)
			n_removed = length(var_ind) - length(var_ind2)
			n_removed>0 && @info "- Removed $n_removed variables that where not found in Model"
			n_missing>0 && @info "- Reconstructed $n_missing missing variables"
			issorted(var_ind2) || @info "- Reordered variables to match Model"
		end
	elseif model.var == :copy
		var = copy(data.var)
	elseif !isempty(model.scaling) && model.annotate
		var = copy(data.var; copycols=false) # keep columns, but create new DataFrame so we can add column below
	else
		var = data.var # actually keep
	end

	# negβT has the variable order of the model, so we add this term after reordering
	if !isempty(negβT) # empty if no intercept and no covariates
		matrix = matrixsum(_named_matrix(matrix,:A), matrixproduct(Symbol("(-β)")=>negβT, :X=>design.matrix'))
	end

	if !isempty(model.scaling)
		matrix = matrixproduct(:D=>Diagonal(model.scaling), _named_matrix(matrix,:A))
		if model.annotate
			var.scaling = copy(model.scaling)
		end
	end

	update_matrix(data, matrix, model; var, model.obs)
end
function project_impl(data::DataMatrix, model::NormalizationModel; external_obs=nothing, kwargs...)
	value_vectors = [_get_value_vector(data,cov,external_obs) for cov in model.covariates]
	design = designmatrix(data, model.covariates, value_vectors)
	project_impl(data, model, design; kwargs...)
end


"""
	normalize_matrix(data::DataMatrix, design::DesignMatrix; scale=false, kwargs...)

Normalize `data` using the specified `design` matrix.

See also: [`NormalizationModel`](@ref), [`designmatrix`](@ref)
"""
function normalize_matrix(data::DataMatrix, design::DesignMatrix; kwargs...)
	model = NormalizationModel(data, design; kwargs...)
	project_impl(data, model, design)
end

"""
	normalize_matrix(data::DataMatrix, [covariates...]; center=true, scale=false, kwargs...)

Normalize `data`. By default, the matrix is centered.
Any `covariates` specified (using column names of `data.obs`) will be regressed out.

* `center` - Set to true to center the data matrix.
* `scale` - Set to true to scale the variables in the data matrix to unit standard deviation.

For other `kwargs` and more detailed descriptions, see `NormalizationModel` and `designmatrix`.

# Examples
Centering only:
```julia
julia> normalize_matrix(data)
```

Regression model with intercept (centering) and "fraction_mt" (numerical annotation):
```julia
julia> normalize_matrix(data, "fraction_mt")
```

As above, but also including "batch" (categorical annotation):
```julia
julia> normalize_matrix(data, "fraction_mt", "batch")
```

See also: [`NormalizationModel`](@ref), [`designmatrix`](@ref)
"""
function normalize_matrix(data::DataMatrix, args...; center=true, max_categories=nothing, kwargs...)
	design = designmatrix(data, args...; center, max_categories)
	normalize_matrix(data, design; kwargs...)
end


# - show -

Base.show(io::IO, ::InterceptCovariate) = print(io, "1")
Base.show(io::IO, c::NumericalCovariate) = print(io, "num(", c.name, c.external ? " (external)" : "", ')')
Base.show(io::IO, c::CategoricalCovariate) = print(io, "cat(", c.name, c.external ? " (external)" : "", ')')
Base.show(io::IO, ::MIME"text/plain", ::InterceptCovariate) = print(io, "1")
Base.show(io::IO, ::MIME"text/plain", c::NumericalCovariate) = print(io, "num(", c.name, c.external ? " (external)" : "", ')')
Base.show(io::IO, ::MIME"text/plain", c::CategoricalCovariate) = print(io, "cat(", c.name, c.external ? " (external)" : "", ",n=", length(c.values), ')')

function Base.show(io::IO, ::MIME"text/plain", model::NormalizationModel)
	print(io, "NormalizationModel(rank=", model.rank)
	isempty(model.scaling) || print(io, ", scale=true")
	print(io, ", ~")
	join(io, model.covariates, '+')
	print(io, ')')
end
