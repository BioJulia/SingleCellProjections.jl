negative_regression_matrix_impl(::Action, data, dm; kwargs...) =
	cached(create_job(SCPCore.negative_regression_matrix, data, dm; kwargs..., __version=v"0.1.0")) # NB: No action, always use original
negative_regression_matrix_impl_job(data, dm; kwargs...) =
	create_job(Projectable(negative_regression_matrix_impl), data, dm; kwargs...)


function negative_regression_matrix(::Mat, data, dm; kwargs...)
	# TODO: check that data and dm IDs match
	negative_regression_matrix_impl_job(get_matrix_job(data), get_matrix_job(dm); kwargs...)
end
negative_regression_matrix(::Var, data, dm; kwargs...) = get_job(Var(), data)
negative_regression_matrix(::Obs, data, dm; kwargs...) = get_job(Obs(), dm)


function negative_regression_matrix_job(data, dm; rtol=nothing)
	rtol = @something rtol sqrt(eps())
	create_job(DataMatrixFunction(negative_regression_matrix), data, dm; rtol)
end






function normalize_matrix(::Preprocessing, data, args...; center=true,
		variance_col = nothing,
		std_col = nothing,
		relative_std_col = nothing,
		annotate_variance = variance_col !== nothing,
		annotate_std = std_col !== nothing,
		annotate_relative_std = relative_std_col !== nothing,
		kwargs...)
	dm = designmatrix_job(data, args...; center)
	negβT = negative_regression_matrix_job(data, dm; kwargs...)
	dmT = adjoint_job(dm)
	normalized = matrix_sum_job(:A=>data, matrix_product_job(Symbol("(-β)")=>negβT, :X=>dmT))

	if annotate_variance || annotate_std || annotate_relative_std
		center || throw(ArgumentError("Annotating variance/std/relative_std requires center=true (data must be mean-centered);"))
		base = normalized

		if annotate_variance
			variance_col = @something variance_col "variance"
			normalized = SCP.annotate_var(normalized, SCP.variance(base; assume_centered=true, col=variance_col))
		end
		if annotate_std
			std_col = @something std_col "std"
			normalized = SCP.annotate_var(normalized, SCP.std(base; assume_centered=true, col=std_col))
		end
		if annotate_relative_std
			relative_std_col = @something relative_std_col "relative_std"
			normalized = SCP.annotate_var(normalized, SCP.relative_std(base; assume_centered=true, col=relative_std_col))
		end
	end
	normalized
end
