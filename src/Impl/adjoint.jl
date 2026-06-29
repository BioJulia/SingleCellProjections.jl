adjoint_matrix_job(X) = create_job(LinearAlgebra.adjoint, X; __version=v"0.1.0")

# TODO: Should we do unwrapping of adjoint(adjoint(X)) should probably be done as a late preprocessing step.
function adjoint(::Mat, data)
	if data.f == DataMatrixFunction(adjoint)
		SCP.get_matrix(data.args[1]) # adjoint(adjoint(X)) == X
	else
		adjoint_matrix_job(SCP.get_matrix(data))
	end
end
adjoint(::Var, data) = SCP.get_obs(data)
adjoint(::Obs, data) = SCP.get_var(data)
