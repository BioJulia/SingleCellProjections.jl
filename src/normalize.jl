negative_regression_matrix_impl(::Action, data, dm; kwargs...) =
	cached(create_spec(SCPCore.negative_regression_matrix, data, dm; kwargs..., __version=v"0.1.0")) # NB: No action, always use original
negative_regression_matrix_impl_spec(data, dm; kwargs...) =
	create_spec(Projectable(negative_regression_matrix_impl), data, dm; kwargs...)


function negative_regression_matrix(::Mat, data, dm; kwargs...)
	# TODO: check that data and dm IDs match
	negative_regression_matrix_impl_spec(get_matrix_spec(data), get_matrix_spec(dm); kwargs...)
end
negative_regression_matrix(::Var, data, dm; kwargs...) = get_spec(Var(), data)
negative_regression_matrix(::Obs, data, dm; kwargs...) = get_spec(Obs(), dm)


function negative_regression_matrix_spec(data, dm; rtol=nothing)
	rtol = @something rtol sqrt(eps())
	create_spec(DataMatrixFunction(negative_regression_matrix), data, dm; rtol)
end
function Jobs.negative_regression_matrix(args...; kwargs...)
	negative_regression_matrix_spec(args...; kwargs...)
end






function normalize_matrix(::Preprocessing, data, args...; center=true,
		variance_col = nothing,
		std_col = nothing,
		relative_std_col = nothing,
		annotate_variance = variance_col !== nothing,
		annotate_std = std_col !== nothing,
		annotate_relative_std = relative_std_col !== nothing,
		kwargs...)
	dm = designmatrix_spec(data, args...; center)
	negβT = negative_regression_matrix_spec(data, dm; kwargs...)
	dmT = adjoint_spec(dm)
	normalized = matrix_sum_spec(:A=>data, matrix_product_spec(Symbol("(-β)")=>negβT, :X=>dmT))

	if annotate_variance || annotate_std || annotate_relative_std
		center || throw(ArgumentError("Annotating variance/std/relative_std requires center=true (data must be mean-centered);"))
		base = normalized

		if annotate_variance
			variance_col = @something variance_col "variance"
			normalized = Jobs.annotate_var(normalized, Jobs.variance(base; assume_centered=true, col=variance_col))
		end
		if annotate_std
			std_col = @something std_col "std"
			normalized = Jobs.annotate_var(normalized, Jobs.std(base; assume_centered=true, col=std_col))
		end
		if annotate_relative_std
			relative_std_col = @something relative_std_col "relative_std"
			normalized = Jobs.annotate_var(normalized, Jobs.relative_std(base; assume_centered=true, col=relative_std_col))
		end
	end
	normalized
end


"""
	Jobs.normalize_matrix(data, args...; kwargs...)
"""
function Jobs.normalize_matrix(data, args...; kwargs...)
	create_spec(Preprocess(normalize_matrix), data, args...; kwargs...)
end
