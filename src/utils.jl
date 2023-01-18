"""
	splitrange(r::UnitRange, nparts::Integer)

Splits a range in `nparts` number of parts of equal length.
"""
function splitrange(r::UnitRange{T}, nbrParts::Integer) where T<:Real
	s = first(r)
	d,r = divrem(length(r),nbrParts)
	out = Vector{UnitRange{T}}(undef, nbrParts)
	for i=1:nbrParts
		len = d+(i<=r)
		out[i] = range(s, length=len)
		s += len
	end
	out
end

function index2matrix(ind::Vector{<:Union{Nothing,Int}}, nrow::Integer)
	ncol = length(ind)
	ind2 = findall(!isnothing, ind)
	ind = Int.(ind[ind2])
	sparse(ind,ind2,true,nrow,ncol)
end
