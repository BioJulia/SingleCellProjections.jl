# TODO: generalize to handle both VarAnnotationModel and ObsAnnotationModel with the same struct
struct ObsAnnotationModel <: ProjectionModel
	var_match::DataFrame
	out_names::Vector{String}
	var::Symbol
	obs::Symbol
	matrix::Symbol
end

_get_name_src(fvar::Pair) = first(fvar)
_get_name_src(::Any) = nothing

_default_out_name(row::DataFrameRow, id_cols::AbstractVector, delim) = join([row[col] for col in id_cols], delim)
_default_out_name(row::DataFrameRow, id_col, ::Any) = row[id_col]
_default_out_name(id_cols, delim='_') = row->_default_out_name(row, id_cols, delim)

function instantiate_out_names(var::DataFrame, out_names::AbstractVector)
	length(out_names)==size(var,1) || throw(ArgumentError("Expected out_names ($out_names) to have the same length as the number of matched annotations."))
	out_names
end
instantiate_out_names(var::DataFrame, out_names::Union{AbstractString,Symbol}) = instantiate_out_names(var,[out_names])
instantiate_out_names(var::DataFrame, f) = f.(eachrow(var))

function ObsAnnotationModel(fvar, data::DataMatrix;
                            name_src=nothing, names=nothing,
                            var=:keep, obs=:keep, matrix=:keep)
	@assert var in (:keep, :copy)
	@assert obs in (:keep, :copy)
	@assert matrix in (:keep, :copy)

	names !== nothing && name_src !== nothing && throw(ArgumentError("At most one of `name_src` and `names` should be specified."))

	# kwargs trick to let defaults be decided here if `nothing` is passed to name_src or names
	if names === nothing
		name_src = @something name_src _get_name_src(fvar) Base.names(data.var,1)
		names = _default_out_name(name_src)
	end

	var_ind = _filter_indices(data.var, fvar)
	v = data.var[var_ind,:]
	var_match = select(v, 1; copycols=false)
	isempty(var_match) && throw(ArgumentError("No variables match filter ($fvar)."))
	ObsAnnotationModel(var_match, instantiate_out_names(v, names), var, obs, matrix)
end


projection_isequal(m1::ObsAnnotationModel, m2::ObsAnnotationModel) =
	m1.var_match == m2.var_match && m1.out_names == m2.out_names

# TODO: support updating out_names?
update_model(m::ObsAnnotationModel; var=m.var, obs=m.obs, matrix=m.matrix, kwargs...) =
	(ObsAnnotationModel(m.var_match, m.out_names, var, obs, matrix), kwargs)


function _new_annot(data::DataMatrix, model::ObsAnnotationModel; verbose=false)
	var_ind = table_indexin(model.var_match, data.var; cols=names(model.var_match))

	# nothings in var_ind result in columns with missing values in output
	missing_var = var_ind .== nothing
	verbose && any(missing_var) && @info "$(join(model.out_names[missing_var],", "," and ")) missing in DataMatrix."
	var_ind = var_ind[.!missing_var]

	S = sparse(1:length(var_ind),var_ind,true,length(var_ind),size(data,1)) # TODO: merge code with ind2sparse?
	values = (S*data.matrix)'
	values = convert(Matrix, values)

	# insert columns with missing here
	columns = Vector{Union{Vector{eltype(values)},Vector{Union{eltype(values),Missing}}}}(undef, length(missing_var))
	columns[.!missing_var] = [values[:,j] for j=1:length(var_ind)]
	columns[missing_var] = [missings(eltype(data.matrix), size(data,2)) for j=1:count(missing_var)]
	DataFrame(columns, model.out_names)
end

function var_to_obs!(fvar, data; kwargs...)
	model = ObsAnnotationModel(fvar, data; kwargs...)
	new_obs = _new_annot(data, model)
	insertcols!(data.obs,pairs(eachcol(new_obs))...)
	push!(data.models, model)
	data
end

function var_to_obs(fvar, data; name_src=nothing, names=nothing,
                                var=:copy, obs=:copy, matrix=:keep, kwargs...)
	model = ObsAnnotationModel(fvar, data; name_src, names, var, obs, matrix)
	project_impl(data, model; kwargs...)
end

function var_to_obs_table(fvar, data; kwargs...)
	model = ObsAnnotationModel(fvar, data; kwargs...)
	new_obs = _new_annot(data, model)
	hcat(select(data.obs, 1), new_obs)
end




function project_impl(data::DataMatrix, model::ObsAnnotationModel; verbose=true, kwargs...)
	obs = data.obs
	model.obs == :copy && (obs = copy(obs))

	new_obs = _new_annot(data, model; verbose)

	# TODO: cleanup code
	for name in names(new_obs)
		insertcols!(obs, name=>new_obs[!,name])
	end

	matrix = model.matrix == :keep ? data.matrix : copy(data.matrix)
	update_matrix(data, matrix, model; model.var, obs)
end


# - show -
function Base.show(io::IO, ::MIME"text/plain", model::ObsAnnotationModel)
	print(io, "ObsAnnotationModel(")
	join(io, model.out_names, ", ")
	print(io, ')')
end
