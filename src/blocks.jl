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


# Convenience function for applying a function to each block in a hblock spec (or to a single spec that is not wrapped in hblock)
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



# Somewhat experimental solution for extracting ranges (so that we can match the blocking elsewhere)
function combine_block_ranges(outer_ranges::AbstractVector{T}, inner_ranges) where T
	@assert length(outer_ranges) == length(inner_ranges)
	@assert length.(outer_ranges) == last.(last.(inner_ranges))
	inner_ranges = map((br,ir)-> (.+).(first(br)-1,ir), outer_ranges, inner_ranges) # TODO: Make more readable
	reduce(vcat, inner_ranges)
end
combine_block_ranges_spec(outer_ranges, inner_ranges) =
	create_spec(combine_block_ranges, outer_ranges, inner_ranges; __version=v"0.0.1")


get_col_ranges_impl_spec(A) =
	create_spec(SCPCore.get_col_ranges, A; __version=v"0.0.1")

function get_col_ranges_pre(::Preprocessing, A)
	if is_hblock(A)
		matrices = A.args[1]
		outer_ranges = A.kwargs[:ranges]
		inner_ranges = get_col_ranges_impl_spec.(matrices)
		combine_block_ranges_spec(outer_ranges, inner_ranges)
	else
		get_col_ranges_impl_spec(A)
	end
end

get_col_ranges_spec(A) =
	create_spec(Preprocess{false}(get_col_ranges_pre), A)



get_row_ranges_impl_spec(A) =
	create_spec(SCPCore.get_row_ranges, A; __version=v"0.0.1")

function get_row_ranges_pre(::Preprocessing, A)
	if is_hblock(A)
		get_row_ranges_impl_spec(first(A.args[1])) # row ranges are expected to match for all matrices in the hblock, just use ranges from the first
	else
		get_row_ranges_impl_spec(A)
	end
end

get_row_ranges_spec(A) =
	create_spec(Preprocess{false}(get_row_ranges_pre), A)
