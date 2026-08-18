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


"""
	kwargs_of(f, argtypes...)

The keyword names accepted by the method of `f` that would be called for `argtypes`.

Used by the extensions to derive a [`check_kwargs`](@ref) allow-list from the third-party function
they forward to, so that the list follows that package's version instead of going stale. Errors if
the method slurps `kwargs...`, since then there is nothing to enumerate.
"""
function kwargs_of(f, argtypes...)
	kw = Base.kwarg_decl(which(f, Tuple{argtypes...}))
	any(k->endswith(String(k), "..."), kw) &&
		error("$f($(join(argtypes, ", "))) slurps keyword arguments, cannot derive an allow-list.")
	Tuple(kw)
end
