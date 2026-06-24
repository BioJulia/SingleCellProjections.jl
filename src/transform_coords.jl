"""
    rot2d(α)

Return a 2×2 rotation matrix for angle `α` (in radians). For use with [`Jobs.transform_coords`](@ref).
"""
function rot2d(α)
	s,c = sincos(α)
	[c -s; s c]
end

"""
    flipx2d()

Return a 2×2 matrix that flips the x-axis. For use with [`Jobs.transform_coords`](@ref).
"""
flipx2d() = [-1 0; 0  1]

"""
    flipy2d()

Return a 2×2 matrix that flips the y-axis. For use with [`Jobs.transform_coords`](@ref).
"""
flipy2d() = [ 1 0; 0 -1]


"""
    rotx(α)

Return a 3×3 rotation matrix around the x-axis by angle `α` (in radians). For use with [`Jobs.transform_coords`](@ref).
"""
function rotx(α)
	s,c = sincos(α)
	[1 0 0; 0 c -s; 0 s c]
end

"""
    roty(α)

Return a 3×3 rotation matrix around the y-axis by angle `α` (in radians). For use with [`Jobs.transform_coords`](@ref).
"""
function roty(α)
	s,c = sincos(α)
	[c 0 s; 0 1 0; -s 0 c]
end

"""
    rotz(α)

Return a 3×3 rotation matrix around the z-axis by angle `α` (in radians). For use with [`Jobs.transform_coords`](@ref).
"""
function rotz(α)
	s,c = sincos(α)
	[c -s 0; s c 0; 0 0 1]
end

"""
    flipx3d()

Return a 3×3 matrix that flips the x-axis. For use with [`Jobs.transform_coords`](@ref).
"""
flipx3d() = [-1 0 0; 0  1 0; 0 0  1]

"""
    flipy3d()

Return a 3×3 matrix that flips the y-axis. For use with [`Jobs.transform_coords`](@ref).
"""
flipy3d() = [ 1 0 0; 0 -1 0; 0 0  1]

"""
    flipz3d()

Return a 3×3 matrix that flips the z-axis. For use with [`Jobs.transform_coords`](@ref).
"""
flipz3d() = [ 1 0 0; 0  1 0; 0 0 -1]


function transform_coords_impl(X::TM, transform::TT) where {TM,TT}
	TM <: ROMat && (X = parent(X))
	TT <: ROMat && (transform = parent(transform))

	@assert allequal(size(transform))
	@assert size(X,1) == size(transform,2)
	transform * X
end

transform_coords(::Mat, data, transform; kwargs...) =
	create_job(transform_coords_impl, get_matrix_job(data), transform; __version=v"0.0.1")
function transform_coords(::Var, data, transform; keep_var=false)
	if keep_var
		get_var_job(data)
	else
		prefixed_ids_job("dim_id", "dim", size(transform,1))
	end
end

transform_coords(::Obs, data, transform; kwargs...) = get_obs_job(data)


transform_coords_job(data, transform; kwargs...) =
	create_job(DataMatrixFunction(transform_coords), data, transform; kwargs...)
"""
    Jobs.transform_coords(data, transform; kwargs...) -> Job

Apply a coordinate transformation matrix `transform` to the matrix of `data`.

See also [`Jobs.find_optimal_coord_transform`](@ref), [`Jobs.force_layout`](@ref).
"""
Jobs.transform_coords(data, transform; kwargs...) =
	transform_coords_job(data, transform; kwargs...)



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
	ind_specs = (create_find_matching_ind_job(arg, get_obs_job(data); project_ids=:no) for arg in args)
	create_job(find_optimal_coord_transform_impl, get_matrix_job(data), ind_specs...; kwargs..., __version=v"0.1.0")
end

find_optimal_coord_transform_job(args...; kwargs...) =
	create_job(Projectable(find_optimal_coord_transform), args...; kwargs...)

# Find a better name?
"""
    Jobs.find_optimal_coord_transform(data, group_filters...; kwargs...) -> Job

Find an optimal rotation matrix that aligns `data` coordinates so that specified cell
groups are separated along the principal axes. The first group filter defines the direction
of the first axis (up), the second group the second axis, and so on — each is made orthogonal
to the preceding axes.

Each `group_filter` is a `Pair` of column name and predicate (e.g. `"celltype" => isequal("HSC")`).

# Examples

Rotation of 3D plot:
```julia
julia> transform = Jobs.find_optimal_coord_transform(fl,
           "celltype"=>isequal("HSC"),
           "celltype"=>isequal("T-cells"),
           "celltype"=>isequal("B-cells"))
julia> fl_rotated = Jobs.transform_coords(fl, transform; keep_var=true)
```

Rotation of 2D plot:
```julia
julia> transform = Jobs.find_optimal_coord_transform(fl_2d,
           "celltype"=>isequal("HSC"),
           "celltype"=>isequal("T-cells"))
julia> fl_rotated = Jobs.transform_coords(fl_2d, transform; keep_var=true)
```

See also [`Jobs.transform_coords`](@ref), [`Jobs.force_layout`](@ref).
"""
function Jobs.find_optimal_coord_transform(args...; kwargs...)
	find_optimal_coord_transform_job(args...)
end
