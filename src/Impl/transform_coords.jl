function transform_coords_impl(X::TM, transform::TT) where {TM,TT}
	TM <: ROMat && (X = parent(X))
	TT <: ROMat && (transform = parent(transform))

	@assert allequal(size(transform))
	@assert size(X,1) == size(transform,2)
	transform * X
end

transform_coords(::Mat, data, transform; kwargs...) =
	create_job(transform_coords_impl, SCP.get_matrix(data), transform; __version=v"1.0.0")
function transform_coords(::Var, data, transform; keep_var=false)
	if keep_var
		SCP.get_var(data)
	else
		prefixed_ids_job("dim_id", "dim", size(transform,1))
	end
end

transform_coords(::Obs, data, transform; kwargs...) = SCP.get_obs(data)


_default_transform_axis_order(::Val{2}) = [2,1] # y is up
_default_transform_axis_order(::Val{3}) = [3,1,2] # z is up
_default_transform_axis_order(::Val{N}) where N = nothing # default

function find_optimal_coord_transform_impl(X, indices::T...; order=_default_transform_axis_order(Val(size(X,1)))) where T
	d = size(X,1)
	@assert length(indices) == d
	@assert order === nothing || length(order) == d

	center = mean(X; dims=2)

	centroids = (ind->vec(mean(@view(X[:,ind]); dims=2).-center)).(indices)

	U = zeros(d,d)
	for i in 1:d
		u = centroids[i]

		if i>1 # orthogonalize
			Ui = @view(U[:,1:i-1]) # already fixed directions
			u = u .- Ui*Ui'u
		end

		U[:,i] = u / sqrt(sum(abs2,u))
	end

	order !== nothing && (U[:,order] = U)
	copy(U') # the inverse of a unitary matrix is the adjoint
end
find_optimal_coord_transform_impl(X::ROMat, args...; kwargs...) = find_optimal_coord_transform_impl(parent(X), args...; kwargs...)


function find_optimal_coord_transform(::Action, data, args...; kwargs...)
	# NB: Do not apply action at all, the layout is based on the unprojected data set
	ind_specs = (create_find_matching_ind_job(arg, SCP.get_obs(data); project_ids=:no) for arg in args)
	create_job(find_optimal_coord_transform_impl, SCP.get_matrix(data), ind_specs...; kwargs..., __version=v"1.0.0")
end
