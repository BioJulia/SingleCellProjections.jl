"""
    rot2d(α)

Return a 2×2 rotation matrix for angle `α` (in radians). For use with [`transform_coords`](@ref).
"""
function rot2d(α)
	s,c = sincos(α)
	[c -s; s c]
end

"""
    flipx2d()

Return a 2×2 matrix that flips the x-axis. For use with [`transform_coords`](@ref).
"""
flipx2d() = [-1 0; 0  1]

"""
    flipy2d()

Return a 2×2 matrix that flips the y-axis. For use with [`transform_coords`](@ref).
"""
flipy2d() = [ 1 0; 0 -1]


"""
    rotx(α)

Return a 3×3 rotation matrix around the x-axis by angle `α` (in radians). For use with [`transform_coords`](@ref).
"""
function rotx(α)
	s,c = sincos(α)
	[1 0 0; 0 c -s; 0 s c]
end

"""
    roty(α)

Return a 3×3 rotation matrix around the y-axis by angle `α` (in radians). For use with [`transform_coords`](@ref).
"""
function roty(α)
	s,c = sincos(α)
	[c 0 s; 0 1 0; -s 0 c]
end

"""
    rotz(α)

Return a 3×3 rotation matrix around the z-axis by angle `α` (in radians). For use with [`transform_coords`](@ref).
"""
function rotz(α)
	s,c = sincos(α)
	[c -s 0; s c 0; 0 0 1]
end

"""
    flipx3d()

Return a 3×3 matrix that flips the x-axis. For use with [`transform_coords`](@ref).
"""
flipx3d() = [-1 0 0; 0  1 0; 0 0  1]

"""
    flipy3d()

Return a 3×3 matrix that flips the y-axis. For use with [`transform_coords`](@ref).
"""
flipy3d() = [ 1 0 0; 0 -1 0; 0 0  1]

"""
    flipz3d()

Return a 3×3 matrix that flips the z-axis. For use with [`transform_coords`](@ref).
"""
flipz3d() = [ 1 0 0; 0  1 0; 0 0 -1]


"""
    SCP.transform_coords(data, transform; kwargs...) -> Job

Apply a coordinate transformation matrix `transform` to the matrix of `data`.

See also [`find_optimal_coord_transform`](@ref), [`force_layout`](@ref).
"""
transform_coords(data, transform; kwargs...) =
	Impl.transform_coords_job(data, transform; kwargs...)


# Find a better name?
"""
    SCP.find_optimal_coord_transform(data, group_filters...; kwargs...) -> Job

Find an optimal rotation matrix that aligns `data` coordinates so that specified cell
groups are separated along the principal axes. The first group filter defines the direction
of the first axis (up), the second group the second axis, and so on — each is made orthogonal
to the preceding axes.

Each `group_filter` is a `Pair` of column name and predicate (e.g. `"celltype" => isequal("HSC")`).

# Examples

Rotation of 3D plot:
```julia
julia> transform = SCP.find_optimal_coord_transform(fl,
           "celltype"=>isequal("HSC"),
           "celltype"=>isequal("T-cells"),
           "celltype"=>isequal("B-cells"))
julia> fl_rotated = SCP.transform_coords(fl, transform; keep_var=true)
```

Rotation of 2D plot:
```julia
julia> transform = SCP.find_optimal_coord_transform(fl_2d,
           "celltype"=>isequal("HSC"),
           "celltype"=>isequal("T-cells"))
julia> fl_rotated = SCP.transform_coords(fl_2d, transform; keep_var=true)
```

See also [`transform_coords`](@ref), [`force_layout`](@ref).
"""
function find_optimal_coord_transform(args...; kwargs...)
	Impl.find_optimal_coord_transform_job(args...)
end
