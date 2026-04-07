hblock_spec(a) = create_spec(SCPCore.hblock, a; __version=v"0.0.1")
# Jobs.hblock(args...) = create_hblock_spec(args...)

is_hblock(x::SpecUnion) = x.f == SCPCore.hblock
is_hblock(::Any) = false


# Experimental
function hblock_map(f, spec; wrap=hblock_spec)
	if is_hblock(spec)
		wrap([f(x) for x in spec.args[1]]) # NB: this strips any wrapping like Prefetch
	else
		f(spec)
	end
end



# TODO: Naming etc
function blockify_matrix(::Preprocessing, A; kwargs...)
	hblock_map(A) do x
		create_spec(SCPCore.blockify, x; kwargs..., __version=v"0.0.1")
	end
end

blockify_matrix_spec(A; kwargs...) = create_spec(Preprocess{false}(blockify_matrix), A; kwargs...)

blockify(::Mat, data; kwargs...) = blockify_matrix_spec(get_matrix_spec(data); kwargs...)
blockify(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)

blockify_spec(data; kwargs...) = create_spec(DataMatrixFunction(blockify), data; kwargs...)

