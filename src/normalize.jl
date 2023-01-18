
"""
	variable_var(data::DataMatrix)

Computes the variance of each variable in `data`.

!!! note
	`data` must be mean-centered. E.g. by using `normalize_matrix` before calling `variable_var`.

See also: [`variable_std`](@ref), [`normalize_matrix`](@ref)
"""
function variable_var(data::DataMatrix)
	X = matrixexpression(data.matrix)
	d = DiagGram(X') # Assumes X is mean-centered
	v = compute(d)
	v ./= size(X,2)-1
	max.(0.0, v)
end

"""
	variable_std(data::DataMatrix)

Computes the variance of each variable in `data`.

!!! note
	`data` must be mean-centered. E.g. by using `normalize_matrix` before calling `variable_std`.

See also: [`variable_var`](@ref), [`normalize_matrix`](@ref)
"""
variable_std(data::DataMatrix) = sqrt.(variable_var(data))


abstract type AbstractCovariate end
struct InterceptCovariate <: AbstractCovariate end
struct NumericalCovariate <: AbstractCovariate
	name::String
	mean::Float64
	scale::Float64
end
function NumericalCovariate(data::DataMatrix, name::String, center::Bool)
	v = data.obs[!,name]
	@assert all(!ismissing, v) "Missing values not supported for numerical covariates."
	@assert all(!isnan, v) "NaN values not supported for numerical covariates."
	@assert all(!isinf, v) "Inf values not supported for numerical covariates."
	m = center ? mean(v) : 0.0
	s = max(1e-6, maximum(x->abs(x-m), v)) # Avoid scaling up if values are too small in absolute numbers
	NumericalCovariate(name, m, s)
end
struct CategoricalCovariate{T} <: AbstractCovariate
	name::String
	values::Vector{T}
end
CategoricalCovariate(data::DataMatrix, name::String) = CategoricalCovariate(name, unique(data.obs[!,name]))

_length(::AbstractCovariate) = 1
_length(c::CategoricalCovariate) = length(c.values)


struct CovariateDesc
	name::String
	type::Symbol
	function CovariateDesc(name::String, type::Symbol)
		@assert type in (:auto, :numerical, :categorical, :intercept)
		new(name,type)
	end
end
covariate(name::String, type=:auto) = CovariateDesc(name, type)


function instantiate_covariate(data::DataMatrix, c::CovariateDesc, center::Bool)
	t = c.type
	if t == :auto
		t = eltype(data.obs[!,c.name]) <: Union{Missing,Number} ? :numerical : :categorical
	end

	if t == :numerical
		NumericalCovariate(data, c.name, center)
	elseif t == :categorical
		CategoricalCovariate(data, c.name)
	elseif t == :intercept
		InterceptCovariate()
	else
		error("Unknown covariate type.")
	end
end
instantiate_covariate(data::DataMatrix, s::String, center::Bool) = instantiate_covariate(data, covariate(s), center)


function setup_covariates(data::DataMatrix, args...; center=true)
	# @show args
	covariates = AbstractCovariate[]

	center |= any(==(InterceptCovariate()), args)

	if center
		push!(covariates, InterceptCovariate())
	end
	for x in args
		x == InterceptCovariate() && continue # intercept already handled above
		push!(covariates, instantiate_covariate(data, x, center))
	end
	covariates
end


struct DesignMatrix
	matrix::Matrix{Float64}
	covariates::Vector{AbstractCovariate}
	obs_match::DataFrame
end

covariate_design!(A, data, ::InterceptCovariate) = A .= 1.0
function covariate_design!(A, data, c::NumericalCovariate)
	v = data.obs[!,c.name]
	@assert all(!ismissing, v) "Missing values not supported for numerical covariates."
	@assert all(!isnan, v) "NaN values not supported for numerical covariates."
	@assert all(!isinf, v) "Inf values not supported for numerical covariates."

	# Center and scale for numerical stability
	A .= (v .- c.mean)./c.scale
end
function covariate_design!(A, data, c::CategoricalCovariate)
	v = data.obs[!,c.name]

	new_values = setdiff(unique(v), c.values)
	isempty(new_values) || error("Categorical covariate ", c.name, " has values not present in the model: ", join(new_value, ','))

	A .= isequal.(v, permutedims(c.values))
end

function designmatrix(data::DataMatrix, covariates::AbstractVector{<:AbstractCovariate}; max_categories=nothing)
	# This little trick is to get the same default parameters through different code paths
	max_categories===nothing && (max_categories=100)
	max_categories::Int

	C = sum(_length, covariates; init=0)
	N = size(data,2)

	A = zeros(N,C)
	i = 1
	for c in covariates
		len = _length(c)
		len > max_categories && error(len, " categories in categorical variable, was this intended? Change max_categories (", max_categories, ") if you want to increase the number of allowed categories.")
		covariate_design!(view(A,:,i:i+len-1), data, c)
		i += len
	end

	DesignMatrix(A, covariates, select(data.obs, data.obs_id_cols))
end
function designmatrix(data::DataMatrix, args...; center=true, kwargs...)
	covariates = setup_covariates(data, args...; center)
	designmatrix(data, covariates; kwargs...)
end


struct NormalizationModel <: ProjectionModel
	negβT::Matrix{Float64}
	covariates::Vector{AbstractCovariate}
	rank::Int # Just for show
	var_match::DataFrame
	scaling::Vector{Float64} # empty vector means no scaling
	annotate::Bool # true: add std as a var annotation
	var::Symbol
	obs::Symbol
end

function projection_isequal(m1::NormalizationModel, m2::NormalizationModel)
	m1.negβT === m2.negβT && m1.covariates == m2.covariates && m1.rank == m2.rank &&
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

	model = NormalizationModel(negβT, design.covariates, rank, select(data.var,data.var_id_cols), [], annotate, var, obs)
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

function project_impl(data::DataMatrix, model::NormalizationModel, design::DesignMatrix; verbose=true)
	@assert model.var in (:keep, :copy)
	@assert data.var_id_cols == names(model.var_match)

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
			var.std = copy(model.scaling)
		end
	end

	update_matrix(data, matrix, model; var, model.obs)
end
project_impl(data::DataMatrix, model::NormalizationModel; kwargs...) = project_impl(data, model, designmatrix(data, model.covariates); kwargs...)


function normalize_matrix(data::DataMatrix, design::DesignMatrix; kwargs...)
	model = NormalizationModel(data, design; kwargs...)
	project_impl(data, model, design)
end
function normalize_matrix(data::DataMatrix, args...; center=true, max_categories=nothing, kwargs...)
	design = designmatrix(data, args...; center, max_categories)
	normalize_matrix(data, design; kwargs...)
end


# - show -

Base.show(io::IO, ::InterceptCovariate) = print(io, "1")
Base.show(io::IO, c::NumericalCovariate) = print(io, "num(", c.name, ')')
Base.show(io::IO, c::CategoricalCovariate) = print(io, "cat(", c.name, ')')
Base.show(io::IO, ::MIME"text/plain", ::InterceptCovariate) = print(io, "1")
Base.show(io::IO, ::MIME"text/plain", c::NumericalCovariate) = print(io, "num(", c.name, ')')
Base.show(io::IO, ::MIME"text/plain", c::CategoricalCovariate) = print(io, "cat(", c.name, ",n=", length(c.values), ')')

function Base.show(io::IO, ::MIME"text/plain", model::NormalizationModel)
	print(io, "NormalizationModel(rank=", model.rank)
	isempty(model.scaling) || print(io, ", scale=true")
	print(io, ", ~")
	join(io, model.covariates, '+')
	print(io, ')')
end
