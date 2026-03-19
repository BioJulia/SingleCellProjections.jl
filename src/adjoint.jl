# TODO: Should we do unwrapping of adjoint(adjoint(X)) here as well? It's probably a better place.
adjoint_matrix_spec(X) = create_spec(LinearAlgebra.adjoint, X; __version=v"0.1.0")

function adjoint_impl(::Mat, data)
	if data.f == DataMatrixFunction(adjoint_impl)
		get_matrix_spec(data.args[1]) # adjoint(adjoint(X)) == X
	else
		adjoint_matrix_spec(get_matrix_spec(data))
	end
end
adjoint_impl(::Var, data) = get_obs_spec(data)
adjoint_impl(::Obs, data) = get_var_spec(data)

adjoint_spec(data) = create_spec(DataMatrixFunction(adjoint_impl), data)

# NB: We call it transpose even though we use adjoint internally.
#     Because a user is more likely to use data' than transpose(data) even when they mean transposing.
function Jobs.transpose(data)
	adjoint_spec(data)
end
