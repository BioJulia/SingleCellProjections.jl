# For each interval in x, find the point in xValuesSorted closest to the midpoint.
# Ignores points already present in x.
function midpoints(xInd::Vector{Int}, xValuesSorted::Vector{Float64})
	N = length(xInd)
	# midPoints = Float64[]
	midPointInd = Int[] # index in vector of sorted values
	interpInd = Int[]   # index in (the smaller) vector of current values
	interpWeight = Float64[]
	for i=1:N-1
		xInd1 = xInd[i]
		xInd2 = xInd[i+1]
		x1 = xValuesSorted[xInd1]
		x2 = xValuesSorted[xInd2]
		mTarget = 0.5*(x1+x2)
		r = searchsorted(xValuesSorted,mTarget) # TODO: limit search to interval xInd:xInd+1 to improve speed
		# r is of length 1 if there is a unique exact match
		# r is of length 0 if there is no exact match
		@assert length(r)<=1

		# NB: mInd1<=mInd2 (because of how searchsorted works when no match)
		mInd1 = last(r)
		mInd2 = first(r)

		mInd = mTarget-xValuesSorted[mInd1] < xValuesSorted[mInd2]-mTarget ? mInd1 : mInd2 # take the closest to the middle
		m = xValuesSorted[mInd]

		# only add if not identical to one of the end points of the interval
		if mInd!=xInd1 && mInd!=xInd2
			# push!(midPoints,m)
			push!(midPointInd,mInd)
			push!(interpInd, i)
			push!(interpWeight, (m-x1)/(x2-x1))
		end
	end
	midPointInd,interpInd,interpWeight
end


function bilinearapproximation(xInd0::Vector{Int},yInd0::Vector{Int},xValuesSorted::Vector{Float64},yValuesSorted::Vector{Float64},f::Function;
                               rtol=1e-3,atol=0)
	@assert all(1 .<= xInd0 .<= length(xValuesSorted))
	@assert all(1 .<= yInd0 .<= length(yValuesSorted))
	@assert issorted(xInd0)
	@assert issorted(yInd0)
	@assert length(xValuesSorted)==length(unique(xValuesSorted))
	@assert length(yValuesSorted)==length(unique(yValuesSorted))

	xCurrInd = copy(xInd0)
	yCurrInd = copy(yInd0)
	vals = f.(xCurrInd',yCurrInd) # values at grid points

	changed = true
	while changed
		changed = false
		# --- subdivide x if needed ---
		# A. get midpoints
		xMidPointInd,xInterpInd,xInterpWeight = midpoints(xCurrInd,xValuesSorted)
		if !isempty(xMidPointInd)
			# B. evaluate at midpoints for every y and compare to linear interpolation
			actual = f.(xMidPointInd',yCurrInd)
			interp = vals[:,xInterpInd].*(1.0.-xInterpWeight') .+ vals[:,xInterpInd.+1].*xInterpWeight'
			keep = any(.!isapprox.(actual,interp;rtol=rtol,atol=atol); dims=1)[:]

			# C. Add new points
			if any(keep)
				changed = true
				# Merge the two sorted point lists and the values at grid points (TODO: can be done more efficiently.)
				append!(xCurrInd,xMidPointInd[keep])
				vals = hcat(vals,actual[:,keep])
				ind = sortperm(xCurrInd)
				xCurrInd = xCurrInd[ind]
				vals = vals[:,ind]
			end
		end

		# --- subdivide x if needed ---
		# A. get midpoints
		yMidPointInd,yInterpInd,yInterpWeight = midpoints(yCurrInd,yValuesSorted)
		if !isempty(yMidPointInd)
			# B. evaluate at midpoints for every x and compare to linear interpolation
			actual = f.(xCurrInd',yMidPointInd)
			interp = vals[yInterpInd,:].*(1.0.-yInterpWeight) .+ vals[yInterpInd.+1,:].*yInterpWeight
			keep = any(.!isapprox.(actual,interp;rtol=rtol,atol=atol); dims=2)[:]

			# C. Add new points
			if any(keep)
				changed = true
				# Merge the two sorted point lists and the values at grid points (TODO: can be done more efficiently.)
				append!(yCurrInd,yMidPointInd[keep])
				vals = vcat(vals,actual[keep,:])
				ind = sortperm(yCurrInd)
				yCurrInd = yCurrInd[ind]
				vals = vals[ind,:]
			end
		end
	end

	vals, xCurrInd, yCurrInd
end


function linearinterpolationmatrix(xValues::Vector{Float64},xNodes::Vector{Float64},transpose=false)
	# construct sparse matrices for each side
	AI = Int[]
	AJ = Int[]
	AV = Float64[]
	for (i,x) in enumerate(xValues)
		r = searchsorted(xNodes,x)
		k1 = last(r)
		k2 = first(r)
		if k1==k2
			push!(AI,k1)
			push!(AJ,i)
			push!(AV,1.0)
		else
			x1 = xNodes[k1]
			x2 = xNodes[k2]
			w = (x-x1)/(x2-x1)
			push!(AI, k1,    k2)
			push!(AJ, i,     i)
			push!(AV, 1.0-w, w)
		end
	end
	transpose ? sparse(AJ,AI,AV) : sparse(AI,AJ,AV)
end
