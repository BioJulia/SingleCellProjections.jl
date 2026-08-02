_map_value(f, p::Pair) = p.first => f(p.second)
_map_value(f, x::Any) = f(x)

_get_value(p::Pair) = p.second
_get_value(x::Any) = x


matrix_product_impl_job(args...) =
	create_job(SCPCore.matrixproduct, args...; __version=v"1.0.0")

# TODO: check that the inner var/obs IDs match!
matrix_product(::Mat, args...) = matrix_product_impl_job(_map_value.(SCP.get_matrix, args)...)
matrix_product(::Var, args...) = SCP.get_var(_get_value(first(args)))
matrix_product(::Obs, args...) = SCP.get_obs(_get_value(last(args)))

function matrix_product_job(arg1, args...)
	create_job(DataMatrixFunction(matrix_product), arg1, args...)
end


matrix_sum_impl_job(args...) =
	create_job(SCPCore.matrixsum, args...; __version=v"1.0.0")

# TODO: check that the inner var/obs IDs match!
matrix_sum(::Mat, args...) = matrix_sum_impl_job(_map_value.(SCP.get_matrix, args)...)
matrix_sum(::Var, args...) = SCP.get_var(_get_value(first(args)))
matrix_sum(::Obs, args...) = SCP.get_obs(_get_value(first(args)))

function matrix_sum_job(arg1, args...)
	create_job(DataMatrixFunction(matrix_sum), arg1, args...)
end


# TODO: Do we need these?
# matrix_ref(name, matrix) = SCPCore.MatrixRef(name, matrix)
# matrix_ref_job(name, matrix) = create_job(matrix_ref, name, matrix; __version=v"1.0.0")
