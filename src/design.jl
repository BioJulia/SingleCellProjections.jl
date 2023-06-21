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

"""
	covariate(name::String, type=:auto)

Create a `covariate` referring to column `name`.
`type` must be one of `:auto`, `:numerical`, `:categorical` and `:intercept`.
`:auto` means auto-detection by checking if the values in the column are numerical or categorical.
`type==:intercept` adds an intercept to the model (in which case the `name` parameter is ignored).

See also: [`designmatrix`](@ref)
"""
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

"""
	designmatrix(data::DataMatrix, [covariates...]; center=true, max_categories=100)

Creates a design matrix from `data.obs` and the given `covariates`.
Covariates can be specied using strings (column name in data.obs), with autodetection of whether the covariate is numerical or categorical, or using the `covariate` function for more control.

* `center` - If `true`, an intercept is added to the design matrix. (Should only be set to `false` in very rare circumstances.)
* `max_categories` - Safety parameter, an error will be thrown if there are too many categories. In this case, it is likely a mistake that the covariate was used as a categorical covariate. Using a very large number of categories is also bad for performance and memory consumption.

# Examples
Centering only:
```julia
julia> designmatrix(data)
```

Regression model with intercept (centering) and "fraction_mt" (numerical annotation):
```julia
julia> designmatrix(data, "fraction_mt")
```

As above, but also including "batch" (categorical annotation):
```julia
julia> designmatrix(data, "fraction_mt", "batch")
```

See also: [`normalize_matrix`](@ref), [`NormalizationModel`](@ref), [`covariate`](@ref)
"""
function designmatrix(data::DataMatrix, args...; center=true, kwargs...)
	covariates = setup_covariates(data, args...; center)
	designmatrix(data, covariates; kwargs...)
end
