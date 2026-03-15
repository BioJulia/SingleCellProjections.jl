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


function transform_coords_impl(X, transform)
	X = X isa ROMat ? parent(X) : X
	transform = transform isa ROMat ? parent(transform) : transform

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
	Job(transform_coords_spec(data, transform; kwargs...))
