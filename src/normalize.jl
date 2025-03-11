# It's confusing with center_matrix, Jobs.center_matrix and SCPCore.center_matrix being different functions having the same name.
# But we'll get rid of it so that doesn't matter!
function center_matrix(action::Action, matrix)
	# model is created from original data
	model = create_spec(SCPCore.CenteringModel2, matrix; use_cache=true, __version=v"0.1.0")
	create_spec(SCPCore.center_project, model, action(matrix); __version=v"0.1.0")
end


function center_matrix(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(center_matrix), matrix_spec; use_cache=false, kwargs...)
end
center_matrix(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)


# TEMP, use this as a simple example for testing out projections and specs
function Jobs.center_matrix(args...; kwargs...)
	Job(create_spec(DataMatrixFunc(center_matrix), args...; use_cache=false, kwargs...))
end






negative_regression_matrix_impl(::Action, data, dm; kwargs...) =
	create_spec(SCPCore.negative_regression_matrix, data, dm; use_cache=true, kwargs..., __version=v"0.1.0") # NB: No action, always use original
negative_regression_matrix_impl_spec(data, dm; kwargs...) =
	create_spec(Projectable(negative_regression_matrix_impl), data, dm; kwargs...)


function negative_regression_matrix(::Mat, data, dm; kwargs...)
	negative_regression_matrix_impl_spec(get_matrix_spec(data), get_matrix_spec(dm); kwargs...)
end
negative_regression_matrix(::Var, data, dm; kwargs...) = get_spec(Var(), data)
negative_regression_matrix(::Obs, data, dm; kwargs...) = get_spec(Obs(), dm)


function negative_regression_matrix_spec(data, dm; rtol=nothing)
	rtol = @something rtol sqrt(eps())
	create_spec(DataMatrixFunc(negative_regression_matrix), data, dm; use_cache=true, rtol)
end
function Jobs.negative_regression_matrix(args...; kwargs...)
	Job(negative_regression_matrix_spec(args...; kwargs...))
end




normalize_matrix_impl(action::Action, data, negβT, dm) =
	create_spec(SCPCore.normalize_matrix2, action(data), action(negβT), action(dm); use_cache=false, __version=v"0.1.0")
normalize_matrix_impl_spec(data, negβT, dm) =
	create_spec(Projectable(normalize_matrix_impl), data, negβT, dm; use_cache=false)

function normalize_matrix(::Mat, data, args...; center=true, rtol=nothing)
	# Maybe we should go for the matrix specs directly?
	dm = designmatrix_spec(data, args...; center)
	negβT = negative_regression_matrix_spec(data, dm; rtol)
	normalize_matrix_impl_spec(get_matrix_spec(data), get_matrix_spec(negβT), get_matrix_spec(dm))
end
normalize_matrix(f::Union{Var,Obs}, data, args...; center=true) = get_spec(f, data)


function Jobs.normalize_matrix(data, args...; center=true, kwargs...)
	Job(create_spec(DataMatrixFunc(normalize_matrix), data, args...; use_cache=false, center, kwargs...))
end
