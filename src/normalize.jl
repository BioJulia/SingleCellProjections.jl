# It's confusing with center_matrix, Jobs.center_matrix and SCPCore.center_matrix being different functions having the same name.
# But we'll get rid of it so that doesn't matter!
function center_matrix(action::Action, matrix)
	# model is created from original data
	model = create_spec(SCPCore.CenteringModel2, matrix; __use_cache=true, __version=v"0.1.0")
	create_spec(SCPCore.center_project, model, action(matrix); __version=v"0.1.0")
end


function center_matrix(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(center_matrix), matrix_spec; kwargs...)
end
center_matrix(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)


# TEMP, use this as a simple example for testing out projections and specs
function Jobs.center_matrix(args...; kwargs...)
	Job(create_spec(DataMatrixFunc(center_matrix), args...; kwargs...))
end






negative_regression_matrix_impl(::Action, data, dm; kwargs...) =
	create_spec(SCPCore.negative_regression_matrix, data, dm; __use_cache=true, kwargs..., __version=v"0.1.0") # NB: No action, always use original
negative_regression_matrix_impl_spec(data, dm; kwargs...) =
	create_spec(Projectable(negative_regression_matrix_impl), data, dm; kwargs...)


function negative_regression_matrix(::Mat, data, dm; kwargs...)
	negative_regression_matrix_impl_spec(get_matrix_spec(data), get_matrix_spec(dm); kwargs...)
end
negative_regression_matrix(::Var, data, dm; kwargs...) = get_spec(Var(), data)
negative_regression_matrix(::Obs, data, dm; kwargs...) = get_spec(Obs(), dm)


function negative_regression_matrix_spec(data, dm; rtol=nothing)
	rtol = @something rtol sqrt(eps())
	create_spec(DataMatrixFunc(negative_regression_matrix), data, dm; rtol)
end
function Jobs.negative_regression_matrix(args...; kwargs...)
	Job(negative_regression_matrix_spec(args...; kwargs...))
end




# normalize_matrix_impl(action::Action, data, negβT, dm) =
# 	create_spec(SCPCore.normalize_matrix2, action(data), action(negβT), action(dm); __use_cache=false, __version=v"0.1.0")
# normalize_matrix_impl_spec(data, negβT, dm) =
# 	create_spec(Projectable(normalize_matrix_impl), data, negβT, dm)

# function normalize_matrix(::Mat, data, args...; center=true, rtol=nothing)
# 	# Maybe we should go for the matrix specs directly?
# 	dm = designmatrix_spec(data, args...; center)
# 	negβT = negative_regression_matrix_spec(data, dm; rtol)
# 	normalize_matrix_impl_spec(get_matrix_spec(data), get_matrix_spec(negβT), get_matrix_spec(dm))
# end
# normalize_matrix(f::Union{Var,Obs}, data, args...; center=true) = get_spec(f, data)


# function Jobs.normalize_matrix(data, args...; center=true, kwargs...)
# 	Job(create_spec(DataMatrixFunc(normalize_matrix), data, args...; center, kwargs...))
# end


# function add_normalization_matrix(::Mat, data, dm, negβT)
# end
# function add_normalization_matrix(::Var, data, dm, negβT)
# end
# function add_normalization_matrix(::Obs, data, dm, negβT)
# end


# TODO: Move product & sum to another file
_map_value(f, p::Pair) = p.first => f(p.second)
_map_value(f, m::ReproducibleJobs.Managed{<:Pair}) = _map_value(f, ReproducibleJobs.unsafe_unmanage(m))
_map_value(f, x::Any) = f(x)

_get_value(p::Pair) = p.second
_get_value(m::ReproducibleJobs.Managed{<:Pair}) = _get_value(ReproducibleJobs.unsafe_unmanage(m))
_get_value(x::Any) = x


matrix_product_impl(args...) = SCPCore.matrixproduct(args...)
matrix_product_projectable(action::Action, args...) =
	create_spec(matrix_product_impl, action.(args)...; __use_cache=true, __version=v"0.1.1")
matrix_product_projectable_spec(args...) =
	create_spec(Projectable(matrix_product_projectable), args...)


# TODO: check that the inner var/obs IDs match!
matrix_product(::Mat, args...) = matrix_product_projectable_spec(_map_value.(get_matrix_spec, args)...)
matrix_product(::Var, args...) = get_var_spec(_get_value(first(args)))
matrix_product(::Obs, args...) = get_obs_spec(_get_value(last(args)))


function matrix_product_spec(args...)
	isempty(args) && throw(ArgumentError("An empty matrix product is not allowed."))
	create_spec(DataMatrixFunc(matrix_product), args...)
end

function matrix_sum_impl(args...)
	SCPCore.matrixsum(args...)
end
matrix_sum_projectable(action::Action, args...) =
	create_spec(matrix_sum_impl, action.(args)...; __use_cache=true, __version=v"0.1.1")
matrix_sum_projectable_spec(args...) =
	create_spec(Projectable(matrix_sum_projectable), args...)


# TODO: check that the inner var/obs IDs match!
matrix_sum(::Mat, args...) = matrix_sum_projectable_spec(_map_value.(get_matrix_spec, args)...)
matrix_sum(::Var, args...) = get_var_spec(_get_value(first(args)))
matrix_sum(::Obs, args...) = get_obs_spec(_get_value(first(args)))


function matrix_sum_spec(args...)
	isempty(args) && throw(ArgumentError("An empty matrix sum is not allowed."))
	create_spec(DataMatrixFunc(matrix_sum), args...)
end



function normalize_matrix_pre(data, args...; center, kwargs...)
	dm = designmatrix_spec(data, args...; center)
	negβT = negative_regression_matrix_spec(data, dm; kwargs...)

	dmT = adjoint_spec(dm)
	matrix_sum_spec(:A=>data, matrix_product_spec(Symbol("(-β)")=>negβT, :X=>dmT))
end


function Jobs.normalize_matrix(data, args...; center=true, kwargs...)
	Job(create_spec(Preprocess(normalize_matrix_pre), data, args...; center, kwargs...))
end
