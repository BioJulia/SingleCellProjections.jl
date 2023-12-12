"""
	PseudoBulkModel <: ProjectionModel

A model used for computing a "pseudo-bulk" representation of a DataMatrix.


See also: [`pseudobulk`](@ref)
"""
struct PseudoBulkModel <: ProjectionModel
	obs_id_cols::Vector{String} # which columns to use when merging
	merged_id::Union{Nothing,String}
	delim::Union{Nothing,Char,String}
	var::Symbol

	function PseudoBulkModel(obs_id_cols, merged_id, delim, var)
		@assert !isempty(obs_id_cols)
		new(obs_id_cols, merged_id, delim, var)
	end
end
PseudoBulkModel(obs_col, args...; merged_id="id", delim='_', var=:copy) =
	PseudoBulkModel(collect(obs_col, args...), merged_id, delim, var)

projection_isequal(m1::PseudoBulkModel, m2::PseudoBulkModel) =
	m1.obs_id_cols == m2.obs_id_cols &&
	m1.merged_id == m2.merged_id &&
	m1.delim == m2.delim

update_model(m::PseudoBulkModel; var=m.var, kwargs...) =
	(PseudoBulkModel(m.obs_id_cols, m.merged_id, m.delim, var), kwargs)


"""
	pseudobulk(data::DataMatrix, obs_col, [additional_columns...]; var=:copy)

Create a new `DataMatrix` by averging over groups, as specified by the categorical annotation `obs_col` (and optionally additional columns).

* `var` - Can be `:copy` (make a copy of source `var`) or `:keep` (share the source `var` object).

# Examples

Create a pseudobulk representation of each sample:
```julia
julia> pseudobulk(transformed; "sampleName")
```

Create a pseudobulk representation for each celltype in each sample:
```julia
julia> pseudobulk(transformed; "sampleName", "celltype")
```
"""
pseudobulk(data::DataMatrix, obs_col, args::String...; kwargs...) =
	project(data, PseudoBulkModel(obs_col, args...; kwargs...))



function project_impl(data::DataMatrix, model::PseudoBulkModel; verbose=true)
	@assert all(in(names(data.obs)), model.obs_id_cols)

	obs = select(data.obs, model.obs_id_cols)
	unique!(obs)
	dropmissing!(obs) # This is the new obs annotation

	# Find out which group each cell belongs to
	obs_ind = table_indexin(data.obs, obs)

	# Create sparse matrix mapping cells to groups
	N = size(data,2)
	S = sparse(1:N, obs_ind, 1.0, N, size(obs,1))

	# Make each column sum to one (so that we take mean for each group)
	rmul!(S, Diagonal(1.0 ./ vec(sum(S; dims=1))))

	X = matrixproduct(_named_matrix(data.matrix,:A), :S=>S)

	# Add merged id if requested
	if model.merged_id !== nothing
		ids = join.(eachrow(obs), model.delim)
		insertcols!(obs, 1, model.merged_id=>ids)
		obs_id_cols = [model.merged_id]
	else
		obs_id_cols = model.obs_id_cols
	end

	update_matrix(data, X, model; model.var, obs, obs_id_cols)
end

# - show -
function Base.show(io::IO, ::MIME"text/plain", model::PseudoBulkModel)
	print(io, "PseudoBulkModel(")
	join(io, model.obs_id_cols, ", ")
	print(io, ')')
end
