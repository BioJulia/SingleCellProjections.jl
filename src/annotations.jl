# NB: Annotations is considered experimental API and thus not exported.
#     It may get breaking changes in minor/patch releases.
struct Annotations
	df::DataFrame # implementation detail, might be changed later. The first column is the ID column, and the name of that column is the name of the axis.
end

get_table(a::Annotations) = getfield(a,:df)


Base.haskey(a::Annotations, name::String) = hasproperty(get_table(a), name)


function Base.get(f::Union{Type,Function}, a::Annotations, column::String)
	df = get_table(a)
	hasproperty(df, column) || return f()
	id_column = names(df, 1)
	cols = only(id_column) == column ? id_column : vcat(id_column,column)
	Annotations(select(df, cols; copycols=false))
end
Base.get(a::Annotations, column::String, default) = get(()->default, a, column)
Base.get(f::Union{Type,Function}, a::Annotations, column::Symbol) = get(f, a, String(column))
Base.get(a::Annotations, column::Symbol, default) = get(a,String(column), default)


Base.getindex(a::Annotations, column::Union{Symbol,String}) = get(()->throw(KeyError(column)), a, column)

function Base.getindex(a::Annotations, columns::AbstractVector{String})
	df = get_table(a)
	for column in columns
		hasproperty(df, column) || throw(KeyError(column))
	end

	id_column = names(df,1)
	id_ind = findfirst(isequal(only(id_column)), columns)
	if id_ind !== nothing # ID column present? Move it first and keep the relative order between the others.
		cols = append!(id_column, @view(columns[1:id_ind-1]))
		cols = append!(id_column, @view(columns[id_ind+1:end]))
	else # ID column not present? Add it to the beginning.
		cols = append!(id_column, columns)
	end
	Annotations(select(df,cols; copycols=false))
end
Base.getindex(a::Annotations, columns::AbstractVector{<:Union{Symbol,String}}) = a[String.(columns)]


Base.propertynames(a::Annotations, private::Bool) = propertynames(get_table(a), private)
Base.getproperty(a::Annotations, column::Symbol) = a[column]
Base.getproperty(a::Annotations, column::String) = a[column]

function annotation_name(a::Annotations)
	df = get_table(a)
	@assert size(df,2) == 2 "Expected annotations object to have an ID column and a single data column, got columns: $(names(df))"
	only(names(df,2))
end

# function annotation_values(a::Annotations)
# 	df = get_table(a)
# 	@assert size(df,2) == 2 "Expected annotations object to have an ID column and a single data column, got columns: $(names(df))"
# 	df[!,2]
# end
