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
