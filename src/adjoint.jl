# TODO: Should we do unwrapping of adjoint(adjoint(X)) here as well? It's probably a better place.
adjoint_matrix_job(X) = create_job(LinearAlgebra.adjoint, X; __version=v"0.1.0")

function adjoint_impl(::Mat, data)
	if data.f == DataMatrixFunction(adjoint_impl)
		get_matrix_job(data.args[1]) # adjoint(adjoint(X)) == X
	else
		adjoint_matrix_job(get_matrix_job(data))
	end
end
adjoint_impl(::Var, data) = get_obs_job(data)
adjoint_impl(::Obs, data) = get_var_job(data)

adjoint_job(data) = create_job(DataMatrixFunction(adjoint_impl), data)

# NB: We call it transpose even though we use adjoint internally.
#     Because a user is more likely to use data' than transpose(data) even when they mean transposing.
function Jobs.transpose(data)
	adjoint_job(data)
end
