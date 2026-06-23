# hblock_job(a, ranges) = create_job(SCPCore.hblock, a; ranges, __version=v"0.0.1")

# new version were we ensure ranges is fetched
hblock_impl_job(a, ranges) = create_job(SCPCore.hblock, a; ranges, __version=v"0.0.1")
# hblock_pre(::Preprocessing, a; ranges) = hblock_impl_job(a, ranges)

function hblock_pre(::Preprocessing, a; ranges)
	# Remove empty ranges here! Can happen if we have e.g. filtered away an entire sample.

	non_empty = .!isempty.(ranges)
	all(non_empty) && return hblock_impl_job(a, ranges)

	n_non_empty = count(non_empty)
	if n_non_empty > 1
		return hblock_impl_job(a[n_non_empty], ranges[n_non_empty])
	else#if n_non_empty <= 1
		ind = @something findfirst(non_empty) 1 # If all ranges are empty, just take the first (empty) block
		return a[ind] # remove hblock since there is only one block
	end
end
hblock_job(a, ranges) = create_job(Preprocess(hblock_pre), a; ranges=fetched(ranges))



is_hblock(x::SpecRef) = x.f == SCPCore.hblock
is_hblock(::Any) = false


# Convenience function for applying a function to each block in a hblock spec (or to a single spec that is not wrapped in hblock)
function hblock_map(f, spec; wrap=hblock_job)
	if is_hblock(spec)
		wrap([f(x) for x in spec.args[1]], _get_kwarg(spec, :ranges)) # NB: this strips any wrapping like Prefetch
	else
		f(spec)
	end
end




# TODO: Naming etc
function blockify_matrix(::Preprocessing, A; kwargs...)
	hblock_map(A) do x
		create_job(SCPCore.blockify, x; kwargs..., __version=v"0.1.0")
	end
end

blockify_matrix_job(A; kwargs...) = create_job(Preprocess{false}(blockify_matrix), A; kwargs...)

blockify(::Mat, data; kwargs...) = blockify_matrix_job(get_matrix_job(data); kwargs...)
blockify(f::Union{Var,Obs}, data; kwargs...) = get_job(f, data)

blockify_job(data; kwargs...) = create_job(DataMatrixFunction(blockify), data; kwargs...)



# Somewhat experimental solution for extracting ranges (so that we can match the blocking elsewhere)
function combine_block_ranges(outer_ranges::AbstractVector{T}, inner_ranges) where T
	@assert length(outer_ranges) == length(inner_ranges)
	@assert length.(outer_ranges) == last.(last.(inner_ranges))
	inner_ranges = map((br,ir)-> (.+).(first(br)-1,ir), outer_ranges, inner_ranges) # TODO: Make more readable
	reduce(vcat, inner_ranges)
end
combine_block_ranges_job(outer_ranges, inner_ranges) =
	create_job(combine_block_ranges, outer_ranges, inner_ranges; __version=v"0.0.1")


get_col_ranges_impl_job(A) =
	create_job(SCPCore.get_col_ranges, A; __version=v"0.0.1")

function get_col_ranges_pre(::Preprocessing, A)
	if is_hblock(A)
		matrices = A.args[1]
		outer_ranges = _get_kwarg(A, :ranges)
		inner_ranges = get_col_ranges_impl_job.(matrices)
		combine_block_ranges_job(outer_ranges, inner_ranges)
	else
		get_col_ranges_impl_job(A)
	end
end

get_col_ranges_job(A) =
	create_job(Preprocess{false}(get_col_ranges_pre), A)



get_row_ranges_impl_job(A) =
	create_job(SCPCore.get_row_ranges, A; __version=v"0.0.1")

function get_row_ranges_pre(::Preprocessing, A)
	if is_hblock(A)
		get_row_ranges_impl_job(first(A.args[1])) # row ranges are expected to match for all matrices in the hblock, just use ranges from the first
	else
		get_row_ranges_impl_job(A)
	end
end

get_row_ranges_job(A) =
	create_job(Preprocess{false}(get_row_ranges_pre), A)
