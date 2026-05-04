function rot2d(α)
	s,c = sincos(α)
	[c -s; s c]
end
flipx2d() = [-1 0; 0  1]
flipy2d() = [ 1 0; 0 -1]


function rotx(α)
	s,c = sincos(α)
	[1 0 0; 0 c -s; 0 s c]
end
function roty(α)
	s,c = sincos(α)
	[c 0 s; 0 1 0; -s 0 c]
end
function rotz(α)
	s,c = sincos(α)
	[c -s 0; s c 0; 0 0 1]
end
flipx3d() = [-1 0 0; 0  1 0; 0 0  1]
flipy3d() = [ 1 0 0; 0 -1 0; 0 0  1]
flipz3d() = [ 1 0 0; 0  1 0; 0 0 -1]


function transform_coords_impl(X::TM, transform::TT) where {TM,TT}
	TM <: ROMat && (X = parent(X))
	TT <: ROMat && (transform = parent(transform))

	@assert allequal(size(transform))
	@assert size(X,1) == size(transform,2)
	transform * X
end

transform_coords(::Mat, data, transform; kwargs...) =
	create_spec(transform_coords_impl, get_matrix_spec(data), transform; __version=v"0.0.1")
function transform_coords(::Var, data, transform; keep_var=false)
	if keep_var
		get_var_spec(data)
	else
		prefixed_ids_spec("dim_id", "dim", size(transform,1))
	end
end

transform_coords(::Obs, data, transform; kwargs...) = get_obs_spec(data)


transform_coords_spec(data, transform; kwargs...) =
	create_spec(DataMatrixFunction(transform_coords), data, transform; kwargs...)
Jobs.transform_coords(data, transform; kwargs...) =
	transform_coords_spec(data, transform; kwargs...)



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
	ind_specs = (create_find_matching_ind_spec(arg, get_obs_spec(data); project_ids=:no) for arg in args)
	create_spec(find_optimal_coord_transform_impl, get_matrix_spec(data), ind_specs...; kwargs..., __version=v"0.1.0")
end

find_optimal_coord_transform_spec(args...; kwargs...) =
	create_spec(Projectable(find_optimal_coord_transform), args...; kwargs...)

# Find a better name?
function Jobs.find_optimal_coord_transform(args...; kwargs...)
	find_optimal_coord_transform_spec(args...)
end
