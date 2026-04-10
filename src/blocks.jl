# hblock_spec(a, ranges) = create_spec(SCPCore.hblock, a; ranges, __version=v"0.0.1")

# new version were we ensure ranges is fetched
hblock_impl_spec(a, ranges) = create_spec(SCPCore.hblock, a; ranges, __version=v"0.0.1")
# hblock_pre(::Preprocessing, a; ranges) = hblock_impl_spec(a, ranges)

function hblock_pre(::Preprocessing, a; ranges)
	# Remove empty ranges here! Can happen if we have e.g. filtered away an entire sample.

	non_empty = .!isempty.(ranges)
	all(non_empty) && return hblock_impl_spec(a, ranges)

	n_non_empty = count(non_empty)
	if n_non_empty > 1
		return hblock_impl_spec(a[n_non_empty], ranges[n_non_empty])
	else#if n_non_empty <= 1
		ind = @something findfirst(non_empty) 1 # If all ranges are empty, just take the first (empty) block
		return a[ind] # remove hblock since there is only one block
	end
end
hblock_spec(a, ranges) = create_spec(Preprocess(hblock_pre), a; ranges=fetched(ranges))



is_hblock(x::SpecUnion) = x.f == SCPCore.hblock
is_hblock(::Any) = false


# Experimental
function hblock_map(f, spec; wrap=hblock_spec)
	if is_hblock(spec)
		wrap([f(x) for x in spec.args[1]], spec.kwargs[:ranges]) # NB: this strips any wrapping like Prefetch
	else
		f(spec)
	end
end




# TODO: Naming etc
function blockify_matrix(::Preprocessing, A; kwargs...)
	hblock_map(A) do x
		create_spec(SCPCore.blockify, x; kwargs..., __version=v"0.1.0")
	end
end

blockify_matrix_spec(A; kwargs...) = create_spec(Preprocess{false}(blockify_matrix), A; kwargs...)

blockify(::Mat, data; kwargs...) = blockify_matrix_spec(get_matrix_spec(data); kwargs...)
blockify(f::Union{Var,Obs}, data; kwargs...) = get_spec(f, data)

blockify_spec(data; kwargs...) = create_spec(DataMatrixFunction(blockify), data; kwargs...)
