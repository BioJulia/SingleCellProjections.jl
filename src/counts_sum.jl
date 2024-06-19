struct VarCountsSumModel{T} <: ProjectionModel
	var_match::DataFrame
	f::T
	col::String
	var::Symbol
	obs::Symbol
	matrix::Symbol
end
function VarCountsSumModel(counts::DataMatrix{<:AbstractMatrix},
                           filter, col;
                           f = identity,
                           var=:keep, obs=:keep, matrix=:keep,
                           external_var=nothing,
                           check=true)
	var_annot = counts.var
	ind = external_var !== nothing ? _filter_indices(var_annot, filter, external_var) : _filter_indices(var_annot, filter)

	var_id = select(var_annot, 1)
	var_match = var_id[ind, :]

	check && isempty(var_match) && throw(ArgumentError("No variables match filter ($filter)."))

	VarCountsSumModel(var_match, f, string(col), var, obs, matrix)
end

projection_isequal(m1::VarCountsSumModel, m2::VarCountsSumModel) =
	m1.var_match == m2.var_match && m1.f == m2.f && m1.col == m2.col

update_model(m::VarCountsSumModel; col=m.col, var=m.var, obs=m.obs, matrix=m.matrix, kwargs...) =
	(VarCountsSumModel(m.var_match, m.f, string(col), var, obs, matrix), kwargs)



# TODO: this could also be implemented efficiently for MatrixExpressions - but only if f==identity
function _var_counts_sum(f::F, X::SparseMatrixCSC{T}, var_mask) where {F,T}
	P,N = size(X)
	@assert length(var_mask)==P
	f0 = f(zero(T))
	@assert iszero(f0) "Expected $f(0) to equal 0, got $f0."

	out = zeros(typeof(f0+f0), N) # Is there a better way to get the output type?

	# TODO: Can we make this a bit faster? If it wasn't for the var_mask, it would just be sum(f,X;dims=1)
	R = rowvals(X)
	V = nonzeros(X)
	for j=1:N
		out[j] = sum(nzrange(X,j); init=0) do k
			var_mask[R[k]] ? f(V[k]) : f0
		end
	end
	out	
end


function _var_counts_sum(counts::DataMatrix{<:AbstractMatrix}, model::VarCountsSumModel)
	@assert model.matrix in (:keep, :copy)

	# Find the rows in counts.var matching rows in sub/tot
	var_mask = _matching_var_mask(counts.var, model.var_match)

	# sum
	_var_counts_sum(model.f, counts.matrix, var_mask)
end


"""
	var_counts_sum!([f=identity], counts::DataMatrix, filter, col; check=true, var=:keep, obs=:keep, external_var)

For each observation, compute the sum of counts matching the `filter`.

If `f` is specified, it is applied to each element before summing. (Similar to `sum`.)

kwargs:
* `var` - Use this to set `var` in the `ProjectionModel`.
* `obs` - Use this to set `obs` in the `ProjectionModel`. Note that `counts.obs` is changed in place, regardless of the value of `obs`.
* `external_var` - If given, these annotations are used instead of `data.var` when applying `filter`. NB: The IDs of `external_var` must match IDs in `data.var`.
If `check=true`, an error will be thrown if no variables match the pattern.

For more information on filtering syntax, see examples below and the documentation on [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter).

Examples
=========

Sum all "Gene Expression" counts:
```
var_counts_sum!(counts, "feature_type"=>isequal("Gene Expression"), "totalRNACount")
```

Compute the number of "Gene Expression" variables that are expressed (i.e. nonzero):
```
var_counts_sum!(!iszero, counts, "feature_type"=>isequal("Gene Expression"), "nonzeroRNACount")
```

See also: [`var_counts_sum`](@ref)
"""
function var_counts_sum!(f, counts::DataMatrix{<:AbstractMatrix}, args...; kwargs...)
	model = VarCountsSumModel(counts, args...; f=f, var=:keep, obs=:keep, matrix=:keep, kwargs...)
	counts.obs[!,model.col] = _var_counts_sum(counts, model)
	push!(counts.models, model)
	counts
end

var_counts_sum!(counts::DataMatrix{<:AbstractMatrix}, args...; kwargs...) =
	var_counts_sum!(identity, counts, args...; kwargs...)


"""
	var_counts_sum([f=identity], counts::DataMatrix, filter, col; check=true, var=:keep, obs=:keep, external_var)

For each observation, compute the sum of counts matching the `filter`.

If `f` is specified, it is applied to each element before summing. (Similar to `sum`.)

kwargs:
* `var` - Use this to set `var` in the `ProjectionModel`.
* `obs` - Use this to set `obs` in the `ProjectionModel`. Note that `counts.obs` is changed in place, regardless of the value of `obs`.
* `external_var` - If given, these annotations are used instead of `data.var` when applying `filter`. NB: The IDs of `external_var` must match IDs in `data.var`.
If `check=true`, an error will be thrown if no variables match the pattern.

For more information on filtering syntax, see examples below and the documentation on [`DataFrames.filter`](https://dataframes.juliadata.org/stable/lib/functions/#Base.filter).

Examples
=========

Sum all "Gene Expression" counts:
```
var_counts_sum(counts, "feature_type"=>isequal("Gene Expression"), "totalRNACount")
```

Compute the number of "Gene Expression" variables that are expressed (i.e. nonzero):
```
var_counts_sum(!iszero, counts, "feature_type"=>isequal("Gene Expression"), "nonzeroRNACount")
```

See also: [`var_counts_sum!`](@ref)
"""
function var_counts_sum(f, counts::DataMatrix{<:AbstractMatrix}, args...; kwargs...)
	model = VarCountsSumModel(counts, args...; f=f, var=:copy, obs=:copy, matrix=:keep, kwargs...)
	project(counts, model)
end

var_counts_sum(counts::DataMatrix{<:AbstractMatrix}, args...; kwargs...) =
	var_counts_sum(identity, counts, args...; kwargs...)



function project_impl(counts::DataMatrix{<:AbstractMatrix}, model::VarCountsSumModel; verbose=true)
	s = _var_counts_sum(counts, model)

	matrix = model.matrix == :keep ? counts.matrix : copy(counts.matrix)
	obs = _update_annot(counts.obs, model.obs, size(counts,2))
	obs[!,model.col] = s
	update_matrix(counts, matrix, model; model.var, obs)
end


# - show -
function Base.show(io::IO, ::MIME"text/plain", model::VarCountsSumModel)
	print(io, "VarCountsSumModel(",
	          model.f !== identity ? "f=$(model.f), " : "",
	          "size=", size(model.var_match,1),
		      ", col=\"", model.col, "\")")
end
