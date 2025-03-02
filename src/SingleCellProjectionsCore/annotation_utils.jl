find_annotation(::String, ::Nothing) = nothing
function find_annotation(name::String, df::DataFrame)
	hasproperty(df, name) || return nothing
	select(df, [only(names(df,1)), name]; copycols=false)
end
function find_annotation(name::String, a::Annotations)
	x = get(a, name, nothing)
	x !== nothing ? get_table(x) : nothing
end

function find_annotation(name::String, annot::AbstractVector)
	for a in annot
		x = find_annotation(name, a)
		x !== nothing && return x
	end
	nothing
end
