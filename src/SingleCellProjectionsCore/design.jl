function mean_and_scale(v::AbstractVector; center)
	m = center ? mean(v) : 0.0
	# Consider using `norm` or `std` to rescale instead
	s = max(1e-6, maximum(x->abs(x-m), v)) # Avoid scaling up if values are too small in absolute numbers
	m, s
end


abstract type AbstractCovariateDesc end


struct CategoricalCovariateDesc <: AbstractCovariateDesc end
struct NumericalCovariateDesc <: AbstractCovariateDesc end
struct TwoGroupCovariateDesc{T} <: AbstractCovariateDesc
	group_a::T
	group_b::Union{T,Nothing} # optional
end

categorical_covariate() = CategoricalCovariateDesc()
numerical_covariate() = NumericalCovariateDesc()
twogroup_covariate(group_a, group_b=nothing) = TwoGroupCovariateDesc(group_a, group_b)
