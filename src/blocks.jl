hblock_spec(a) = create_spec(SCPCore.hblock, a; __version=v"0.0.1")
# Jobs.hblock(args...) = create_hblock_spec(args...)

is_hblock(x::SpecUnion) = x.f == SCPCore.hblock
is_hblock(::Any) = false


# Experimental
function hblock_map(f, spec)
	if is_hblock(spec)
		hblock_spec([f(x) for x in spec.args[1]]) # NB: this strips any wrapping like Prefetch
	else
		f(spec)
	end
end
