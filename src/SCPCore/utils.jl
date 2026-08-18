"""
	check_kwargs(kwargs, allowed)

Throw an `ArgumentError` naming every keyword in `kwargs` that is not in `allowed`.

Public functions that accept `kwargs...` and forward them should call this first, listing the
keywords they accept. Without it, a keyword that no downstream function recognizes is silently
swallowed - and since keywords are part of the job spec, a misspelled one changes the hash without
changing the computation, i.e. it invalidates the cache and recomputes the exact same result.

`allowed` should only list the keywords that are not already named in the signature.
The check is resolved at compile time and disappears entirely for valid calls.
"""
function check_kwargs(kwargs, allowed::Symbol...)
	unknown = Base.structdiff(NamedTuple(kwargs), NamedTuple{allowed})
	isempty(unknown) && return nothing
	throw(ArgumentError("Unknown keyword argument(s): $(join(keys(unknown), ", ")). Accepted: $(join(allowed, ", "))."))
end
