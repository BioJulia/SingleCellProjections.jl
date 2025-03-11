# SVD is an example where the model comes after the result. I.e. svd(data) => UΣVᵀ, but the model is just UΣ.
function svd(action::Action, matrix; kwargs...)
	# First SVD of unprojected
	svd_spec = create_spec(SCPC.implicitsvd, matrix; use_cache=true, kwargs..., __version=v"0.1.0")

	if action isa Eval
		return svd_spec
	else# if action isa Projection
		return create_spec(SCPC.svd_project, svd_spec, action(matrix); use_cache=true, __version=v"0.1.0")
	end
end


function svd(::Mat, data; kwargs...)
	matrix_spec = get_matrix_spec(data)
	create_spec(Projectable(svd), matrix_spec; kwargs...)
end
svd(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)

function Jobs.svd(args...; seed=1234, kwargs...)
	Job(create_spec(DataMatrixFunc(svd), args...; seed, kwargs...))
end
