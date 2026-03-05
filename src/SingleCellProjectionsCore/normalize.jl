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




# Normalization without model struct
function negative_regression_matrix(matrix, dm::AbstractMatrix; rtol=sqrt(eps()))
	A = matrix
	X = dm

	# TODO: No need to run svd etc. if there just is an intercept.
	F = svd(X)
	negΣinv = Diagonal([σ>rtol ? -1.0/σ : 0.0 for σ in F.S]) # cutoff for numerical stability
	AU = A*F.U
	(AU*negΣinv)*F.Vt
end
