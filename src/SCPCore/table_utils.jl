function table_indexin(a, b; cols=names(b), kwargs...)
	b = select(b, cols; copycols=false)
	a = select(a, cols; copycols=false)
	b.__index__ .= 1:size(b,1)
	leftjoin!(a,b; on=cols, kwargs...)
	coalesce.(a.__index__, nothing)
end
