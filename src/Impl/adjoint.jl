adjoint_matrix_job(X) = create_job(LinearAlgebra.adjoint, X; __version=v"0.1.0")

# TODO: Should we do unwrapping of adjoint(adjoint(X)) should probably be done as a late preprocessing step.
function adjoint(::Mat, data)
	if data.f == DataMatrixFunction(adjoint)
		get_matrix_job(data.args[1]) # adjoint(adjoint(X)) == X
	else
		adjoint_matrix_job(get_matrix_job(data))
	end
end
adjoint(::Var, data) = get_obs_job(data)
adjoint(::Obs, data) = get_var_job(data)
