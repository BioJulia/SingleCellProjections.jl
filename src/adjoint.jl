
function adjoint_matrix(action::Action, X)
	# TODO: Should we do unwrapping of adjoint(adjoint(X)) here as well?
	create_spec(LinearAlgebra.adjoint, action(X); __use_cache=false, __version=v"0.1.0")
end

create_adjoint_matrix_spec(X) =
	create_spec(Projectable(adjoint_matrix), X)

function adjoint_impl(::Mat, data)
	if data.f == DataMatrixFunc(adjoint_impl)
		get_matrix_spec(data.args[1]) # adjoint(adjoint(X)) == X
	else
		create_adjoint_matrix_spec(get_matrix_spec(data))
	end
end
adjoint_impl(::Var, data) = get_obs_spec(data)
adjoint_impl(::Obs, data) = get_var_spec(data)

# NB: We call it transpose even though we use adjoint internally.
#     Because a user is more likely to use data' than transpose(data) even when they mean transposing.
function Jobs.transpose(data)
	Job(create_spec(DataMatrixFunc(adjoint_impl), data))
end
