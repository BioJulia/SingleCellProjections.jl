# TODO: generalize to handle both VarAnnotationModel and ObsAnnotationModel with the same struct
struct ObsAnnotationModel <: ProjectionModel
	var_match::DataFrame
	out_names::Vector{String}
	var::Symbol
	obs::Symbol
	matrix::Symbol
end

_default_out_name(row::DataFrameRow, id_cols::Vector, delim='_') = join([row[col] for col in id_cols], delim)
_default_out_name(id_cols::Vector, delim='_') = row->_default_out_name(row, id_cols, delim)

function instantiate_out_names(var::DataFrame, out_names::AbstractVector)
	length(out_names)==size(var,1) || throw(ArgumentError("Expected out_names ($out_names) to have the same length as the number of matched annotations."))
	out_names
end
instantiate_out_names(var::DataFrame, f) = f.(eachrow(var))

function ObsAnnotationModel(fvar, data::DataMatrix;
                            out_names=_default_out_name(data.var_id_cols),
                            var=:keep, obs=:keep, matrix=:keep)
	@assert var in (:keep, :copy)
	@assert obs in (:keep, :copy)
	@assert matrix in (:keep, :copy)

	var_ind = _filter_indices(data.var, fvar)
	v = data.var[var_ind,:]
	var_match = select(v, data.var_id_cols; copycols=false)
	isempty(var_match) && throw(ArgumentError("No variables match filter ($fvar)."))
	ObsAnnotationModel(var_match, instantiate_out_names(v, out_names), var, obs, matrix)
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
	columns = fill([], length(missing_var)) # dummy init
	columns[.!missing_var] = [values[:,j] for j=1:length(var_ind)]
	columns[missing_var] = [missings(eltype(data.matrix), size(data,2)) for j=1:count(missing_var)]

	DataFrame(columns, model.out_names)
end

function var_to_obs!(fvar, data; out_names=_default_out_name(data.var_id_cols))
	model = ObsAnnotationModel(fvar, data; out_names)
	new_obs = _new_annot(data, model)

	# TODO: cleanup code
	for name in names(new_obs)
		insertcols!(data.obs, name=>new_obs[!,name])
	end

	push!(data.models, model)
	data
end

function var_to_obs(fvar, data; out_names=_default_out_name(data.var_id_cols),
                                  var=:copy, obs=:copy, matrix=:keep, kwargs...)
	model = ObsAnnotationModel(fvar, data; out_names, var, obs, matrix)
	project_impl(data, model; kwargs...)
end

function var_to_obs_table(fvar, data; out_names=_default_out_name(data.var_id_cols))
	model = ObsAnnotationModel(fvar, data; out_names)
	new_obs = _new_annot(data, model)
	hcat(select(data.obs, data.obs_id_cols), new_obs)
end




function project_impl(data::DataMatrix, model::ObsAnnotationModel; verbose=true)
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
