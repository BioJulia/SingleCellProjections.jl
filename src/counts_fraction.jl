struct VarCountsFractionModel <: ProjectionModel
	var_match_sub::DataFrame
	var_match_tot::DataFrame
	col::String
	var::Symbol
	obs::Symbol
	matrix::Symbol
end
function VarCountsFractionModel(counts::DataMatrix{<:AbstractMatrix{<:Integer}},
                                sub_filter, tot_filter, col;
                                var=:keep, obs=:keep, matrix=:keep,
                                check=true)
	var_annot = counts.var
	sub_ind = _filter_indices(var_annot, sub_filter)
	tot_ind = _filter_indices(var_annot, tot_filter)

	var_id = select(var_annot, counts.var_id_cols)
	var_match_sub = var_id[sub_ind, :]
	var_match_tot = var_id[tot_ind, :]

	check && isempty(var_match_sub) && throw(ArgumentError("No variables match sub_filter ($sub_filter)."))
	check && isempty(var_match_tot) && throw(ArgumentError("No variables match tot_filter ($tot_filter)."))

	# Force sub to be a subset of tot
	ind = table_indexin(var_match_sub, var_match_tot)
	var_match_sub = var_match_sub[ind.!==nothing, :]

	check && isempty(var_match_sub) && throw(ArgumentError("No variables match both sub_filter ($sub_filter) and tot_filter ($tot_filter)."))

	VarCountsFractionModel(var_match_sub, var_match_tot, string(col), var, obs, matrix)
end

projection_isequal(m1::VarCountsFractionModel, m2::VarCountsFractionModel) =
	m1.var_match_sub == m2.var_match_sub && m1.var_match_tot == m2.var_match_tot && m1.col == m2.col

update_model(m::VarCountsFractionModel; col=m.col, var=m.var, obs=m.obs, matrix=m.matrix, kwargs...) =
	(VarCountsFractionModel(m.var_match_sub, m.var_match_tot, string(col), var, obs, matrix), kwargs)


# TODO: make general table utility function?
function _matching_var_mask(v, sub)
	bad_ind = findfirst(isnothing, table_indexin(sub,v; cols=	names(sub)))
	if bad_ind !== nothing
		error("Row with contents (", join(sub[bad_ind,:],","), ") not found in var.")
	end
	table_indexin(v, sub) .!== nothing
end

function _var_counts_fraction(counts::DataMatrix{<:AbstractMatrix{<:Integer}}, model::VarCountsFractionModel)
	@assert names(model.var_match_sub) == names(model.var_match_tot)
	@assert model.matrix in (:keep, :copy)
	
	# Find the rows in counts.var matching rows in sub/tot
	sub_mask = _matching_var_mask(counts.var, model.var_match_sub)
	tot_mask = _matching_var_mask(counts.var, model.var_match_tot)

	@assert all(tot_mask .| .!sub_mask) # All elements in sub_mask must also be in tot_mask

	sub_count = vec(sub_mask'counts.matrix)
	tot_count = vec(tot_mask'counts.matrix)

	sub_count ./ max.(1, tot_count) # avoid div by zero
end


"""
	var_counts_fraction!(counts::DataMatrix, sub_filter, tot_filter; check=true)

For each observation, compute the fraction of counts that match a specific variable pattern.
* `sub_filter` decides which variables are counted.
* `tot_filter` decides which variables to include in the total.
* If `check=true`, an error will be thrown if no variables match the patterns.

For more information on filtering syntax, see examples below and the documentation on DataFrames.filter.

Examples
=========

Compute the fraction of reads in MT- genes, considering only "Gene Expression" features (and not e.g. "Antibody Capture").
```
var_counts_fraction!(counts, "name"=>contains(r"^MT-"), "feature_type"=>isequal("Gene Expression"), col="percent_mt")
```

Compute the fraction of reads in MT- genes, when there is no `feature_type` annotation (i.e. all variables are genes).
```
var_counts_fraction!(counts, "name"=>contains(r"^MT-"), Returns(true), col="percent_mt")
```
"""
function var_counts_fraction!(counts::DataMatrix{<:AbstractMatrix{<:Integer}}, args...)
	model = VarCountsFractionModel(counts, args...; var=:keep, obs=:keep, matrix=:keep)
	counts.obs[!,model.col] = _var_counts_fraction(counts, model)
	push!(counts.models, model)
	counts
end

function project_impl(counts::DataMatrix{<:AbstractMatrix{<:Integer}}, model::VarCountsFractionModel; verbose=true)
	frac = _var_counts_fraction(counts, model)

	matrix = model.matrix == :keep ? counts.matrix : copy(counts.matrix)
	obs = _update_annot(counts.obs, model.obs, size(counts,2))
	obs[!,model.col] = frac
	update_matrix(counts, matrix, model; model.var, obs)
end


# - show -
function Base.show(io::IO, ::MIME"text/plain", model::VarCountsFractionModel)
	print(io, "VarCountsFractionModel(subset_size=", size(model.var_match_sub,1),
		      ", total_size=", size(model.var_match_tot,1),
		      ", col=\"", model.col, "\")")
end
