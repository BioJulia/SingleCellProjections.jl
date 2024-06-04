abstract type AbstractCovariate end
struct InterceptCovariate <: AbstractCovariate end
struct NumericalCovariate <: AbstractCovariate
	name::String
	mean::Float64
	scale::Float64
end
function NumericalCovariate(data::DataMatrix, name::String, center::Bool)
	v = data.obs[!,name]
	any(ismissing, v) && throw(ArgumentError("Missing values not supported for numerical covariates."))
	any(isnan, v) && throw(ArgumentError("NaN values not supported for numerical covariates."))
	any(isinf, v) && throw(ArgumentError("Inf values not supported for numerical covariates."))
	m = center ? mean(v) : 0.0
	s = max(1e-6, maximum(x->abs(x-m), v)) # Avoid scaling up if values are too small in absolute numbers
	NumericalCovariate(name, m, s)
end
struct CategoricalCovariate{T} <: AbstractCovariate
	name::String
	values::Vector{T}
end
function CategoricalCovariate(data::DataMatrix, name::String)
	v = unique(data.obs[!,name])
	any(ismissing, v) && throw(ArgumentError("Missing values not supported for categorical covariates."))
	CategoricalCovariate(name, v)
end
struct TwoGroupCovariate{T} <: AbstractCovariate
	name::String
	group_a::T
	group_b::Union{Nothing,T}
	mean::Float64
end
function TwoGroupCovariate(data::DataMatrix, name::String, group_a, group_b, center::Bool)
	v = data.obs[!,name]
	uv = unique(v)
	any(ismissing, uv) && throw(ArgumentError("Missing values not supported for two-group covariates."))

	if group_a === nothing && group_b === nothing
		length(uv) != 2 && throw(ArgumentError("Column \"$name\" have exactly two groups (found \"$uv\")."))
		group_a,group_b = minmax(uv[1],uv[2]) # Keep order stable
	else
		group_a in uv || throw(ArgumentError("Group A (\"$group_a\") not found in column \"$name\"."))
		if group_b !== nothing
			# only group_a and group_b allowed
			group_b in uv || throw(ArgumentError("Group B (\"$group_b\") not found in column \"$name\"."))
			length(uv) > 2 && throw(ArgumentError("Only two groups allowed in column \"$name\" (found \"$uv\")."))
		end
	end

	m = center ? (1.0./count(isequal(group_a),v)) : 0.0
	TwoGroupCovariate(name, group_a, group_b, m)
end

_length(::AbstractCovariate) = 1
_length(c::CategoricalCovariate) = length(c.values)

_covariate_scale(::AbstractCovariate) = 1.0
_covariate_scale(n::NumericalCovariate) = n.scale


struct CovariateDesc{T}
	type::Symbol
	name::String
	group_a::T
	group_b::Union{Nothing,T}
	function CovariateDesc(type::Symbol, name::String, group_a::T, group_b::Union{Nothing,T}) where T
		@assert type in (:auto, :numerical, :categorical, :twogroup, :intercept)
		new{T}(type, name, group_a, group_b)
	end
end
CovariateDesc(type, name) = CovariateDesc(type, name, nothing, nothing)

function covariate_prefix(c::CovariateDesc{T}, suffix='_') where T
	if c.type == :twogroup
		if c.group_b !== nothing
			return string(c.name, '_', c.group_a, "_vs_", c.group_b, suffix)
		elseif c.group_a !== nothing
			return string(c.name, '_', c.group_a, suffix)
		end
	end
	return string(c.name, suffix)
end

"""
	covariate(name::String, type=:auto)

Create a `covariate` referring to column `name`.
`type` must be one of `:auto`, `:numerical`, `:categorical`, `:twogroup` and `:intercept`.
`:auto` means auto-detection by checking if the values in the column are numerical or categorical.
`type==:intercept` adds an intercept to the model (in which case the `name` parameter is ignored).

See also: [`designmatrix`](@ref)
"""
covariate(name::String, type::Symbol=:auto) = CovariateDesc(type, name)

"""
	covariate(name::String, group_a, [group_b])

Create a two-group `covariate` referring to column `name`, comparing `group_a` to `group_b`.
`group_a` and `group_b` must be values occuring in the column `name`.

If `group_b` is not given, `group_a` will be compared to all other observations.

See also: [`designmatrix`](@ref)
"""
covariate(name::String, group_a, group_b=nothing) = CovariateDesc(:twogroup, name, group_a, group_b)

covariate(c::CovariateDesc) = c


function instantiate_covariate(data::DataMatrix, c::CovariateDesc, center::Bool)
	t = c.type
	if t == :auto
		t = eltype(data.obs[!,c.name]) <: Union{Missing,Number} ? :numerical : :categorical
	end

	if t == :numerical
		NumericalCovariate(data, c.name, center)
	elseif t == :categorical
		CategoricalCovariate(data, c.name)
	elseif t == :twogroup
		TwoGroupCovariate(data, c.name, c.group_a, c.group_b, center)
	elseif t == :intercept
		InterceptCovariate()
	else
		error("Unknown covariate type.")
	end
end


function setup_covariates(data::DataMatrix, args...; center=true)
	covariates = AbstractCovariate[]

	center |= any(==(InterceptCovariate()), args)

	if center
		push!(covariates, InterceptCovariate())
	end
	for x in args
		x == InterceptCovariate() && continue # intercept already handled above
		push!(covariates, instantiate_covariate(data, covariate(x), center))
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
	any(ismissing, v) && throw(ArgumentError("Missing values not supported for numerical covariates."))
	any(isnan, v) && throw(ArgumentError("NaN values not supported for numerical covariates."))
	any(isinf, v) && throw(ArgumentError("Inf values not supported for numerical covariates."))

	# Center and scale for numerical stability
	A .= (v .- c.mean)./c.scale
end
function covariate_design!(A, data, c::CategoricalCovariate)
	v = data.obs[!,c.name]

	new_values = setdiff(unique(v), c.values)
	isempty(new_values) || error("Categorical covariate ", c.name, " has values not present in the model: ", join(new_value, ','))

	A .= isequal.(v, permutedims(c.values))
end
function covariate_design!(A, data, t::TwoGroupCovariate)
	v = data.obs[!,t.name]
	any(ismissing, v) && throw(ArgumentError("Missing values not supported for two-group covariates."))

	if t.group_b !== nothing
		new_values = setdiff(unique(v), (t.group_a, t.group_b))
		isempty(new_values) || throw(ArgumentError(("Two-group covariate ", t.name, " has values not present in the model: ", join(new_value, ','))))
	end

	nA = count(==(t.group_a), v)
	nB = length(v)-nA
	nA == 0 && throw(ArgumentError("No values belong to group A (\"$(t.group_a)\")."))
	if nB == 0
		suffix = t.group_b !== nothing ? string(" (\"", t.group_b, "\")") : ""
		throw(ArgumentError("No values belong to group B$suffix."))
	end

	A .= (v.==t.group_a) .- t.mean
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

	DesignMatrix(A, covariates, select(data.obs, 1))
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
