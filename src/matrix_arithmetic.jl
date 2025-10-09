_map_value(f, p::Pair) = p.first => f(p.second)
_map_value(f, m::ReproducibleJobs.Managed{<:Pair}) = _map_value(f, ReproducibleJobs.unsafe_unmanage(m))
_map_value(f, x::Any) = f(x)

_get_value(p::Pair) = p.second
_get_value(m::ReproducibleJobs.Managed{<:Pair}) = _get_value(ReproducibleJobs.unsafe_unmanage(m))
_get_value(x::Any) = x


matrix_product_impl(args...) = SCPCore.matrixproduct(args...)
matrix_product_projectable(action::Action, args...) =
	create_spec(matrix_product_impl, action.(args)...; __use_cache=false, __version=v"0.1.1")
matrix_product_projectable_spec(args...) =
	create_spec(Projectable(matrix_product_projectable), args...)


# TODO: check that the inner var/obs IDs match!
matrix_product(::Mat, args...) = matrix_product_projectable_spec(_map_value.(get_matrix_spec, args)...)
matrix_product(::Var, args...) = get_var_spec(_get_value(first(args)))
matrix_product(::Obs, args...) = get_obs_spec(_get_value(last(args)))


function matrix_product_spec(args...)
	isempty(args) && throw(ArgumentError("An empty matrix product is not allowed."))
	create_spec(DataMatrixFunction(matrix_product), args...)
end

matrix_sum_impl(args...) = SCPCore.matrixsum(args...)
matrix_sum_projectable(action::Action, args...) =
	create_spec(matrix_sum_impl, action.(args)...; __use_cache=false, __version=v"0.1.1")
matrix_sum_projectable_spec(args...) =
	create_spec(Projectable(matrix_sum_projectable), args...)


# TODO: check that the inner var/obs IDs match!
matrix_sum(::Mat, args...) = matrix_sum_projectable_spec(_map_value.(get_matrix_spec, args)...)
matrix_sum(::Var, args...) = get_var_spec(_get_value(first(args)))
matrix_sum(::Obs, args...) = get_obs_spec(_get_value(first(args)))


function matrix_sum_spec(args...)
	isempty(args) && throw(ArgumentError("An empty matrix sum is not allowed."))
	create_spec(DataMatrixFunction(matrix_sum), args...)
end
