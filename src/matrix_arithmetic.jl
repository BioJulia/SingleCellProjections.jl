_map_value(f, p::Pair) = p.first => f(p.second)
_map_value(f, m::ReproducibleJobs.Managed{<:Pair}) = _map_value(f, ReproducibleJobs.unsafe_unmanage(m))
_map_value(f, x::Any) = f(x)

_get_value(p::Pair) = p.second
_get_value(m::ReproducibleJobs.Managed{<:Pair}) = _get_value(ReproducibleJobs.unsafe_unmanage(m))
_get_value(x::Any) = x


matrix_product_impl_spec(args...) =
	create_spec(SCPCore.matrixproduct, args...; __version=v"0.1.1")

# TODO: check that the inner var/obs IDs match!
matrix_product(::Mat, args...) = matrix_product_impl_spec(_map_value.(get_matrix_spec, args)...)
matrix_product(::Var, args...) = get_var_spec(_get_value(first(args)))
matrix_product(::Obs, args...) = get_obs_spec(_get_value(last(args)))

function matrix_product_spec(arg1, args...)
	create_spec(DataMatrixFunction(matrix_product), arg1, args...)
end


matrix_sum_impl_spec(args...) =
	create_spec(SCPCore.matrixsum, args...; __version=v"0.1.1")

# TODO: check that the inner var/obs IDs match!
matrix_sum(::Mat, args...) = matrix_sum_impl_spec(_map_value.(get_matrix_spec, args)...)
matrix_sum(::Var, args...) = get_var_spec(_get_value(first(args)))
matrix_sum(::Obs, args...) = get_obs_spec(_get_value(first(args)))

function matrix_sum_spec(arg1, args...)
	create_spec(DataMatrixFunction(matrix_sum), arg1, args...)
end
