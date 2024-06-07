find_annotation(::String, ::Nothing) = nothing
function find_annotation(name::String, df::DataFrame)
	hasproperty(df, name) || return nothing
	select(df, [only(names(df,1)), name]; copycols=false)
end
find_annotation(name::String, a::Annotations) = get_table(get(a, name, nothing))

function find_annotation(name::String, annot::AbstractVector)
	for a in annot
		x = find_annotation(name, a)
		x !== nothing && return x
	end
	nothing
end
