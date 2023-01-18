function _detect_var_id_cols(var)
	id_cols = String[]
	push!(id_cols, hasproperty(var, "id") ? "id" : first(names(var)))
	hasproperty(var, :feature_type) && push!(id_cols, "feature_type")
	@assert allunique(id_cols)

	v = select(var, id_cols; copycols=false)
	@assert size(unique(v),1) == size(v,1) "Failed to autodetect unique variable IDs (tried $id_cols)."
	id_cols
end

function _detect_obs_id_cols(obs)
	id_col = hasproperty(obs, "id") ? "id" : first(names(obs))
	@assert allunique(obs[!,id_col]) "Failed to autodetect unique observation IDs (tried \"$id_col\")."
	String[id_col]
end


struct DataMatrix{T,Tv,Ts}
	matrix::T
	var::Tv
	obs::Ts
	var_id_cols::Vector{String}
	obs_id_cols::Vector{String}
	models::Vector{ProjectionModel}
	function DataMatrix(matrix::T, var::Tv, obs::Ts, var_id_cols, obs_id_cols, models) where {T,Tv,Ts}
		P,N = size(matrix)
		p = size(var,1)
		n = size(obs,1)
		if P != p || N != n
			throw(DimensionMismatch(string("Matrix has dimensions (",P,',',N,"), but there are ", p, " variable annotations and ", n, " observation annotations.")))
		end

		var_id_cols = @something var_id_cols _detect_var_id_cols(var)
		obs_id_cols = @something obs_id_cols _detect_obs_id_cols(obs)

		table_validatecols(var, var_id_cols)
		table_validatecols(obs, obs_id_cols)
		table_validateunique(var, var_id_cols)
		table_validateunique(obs, obs_id_cols)
		new{T,Tv,Ts}(matrix, var, obs, var_id_cols, obs_id_cols, models)
	end
end
DataMatrix(matrix, var, obs; var_id_cols=nothing, obs_id_cols=nothing) =
	DataMatrix(matrix, var, obs, var_id_cols, obs_id_cols, ProjectionModel[])
DataMatrix() = DataMatrix(zeros(0,0),DataFrame(id=String[]),DataFrame(id=String[]))

Base.:(==)(a::DataMatrix, b::DataMatrix) = false
function Base.:(==)(a::DataMatrix{T,Tv,Ts}, b::DataMatrix{T,Tv,Ts}) where {T,Tv,Ts}
	all(i->getfield(a,i)==getfield(b,i), 1:nfields(a))
end


Base.size(data::DataMatrix) = size(data.matrix)
Base.size(data::DataMatrix, dim::Integer) = size(data.matrix, dim)

Base.axes(data::DataMatrix, d::Integer) = axes(data)[d] # needed to make end work in getindex


var_coordinates(data::DataMatrix{<:Union{SVD,LowRank}}) = var_coordinates(data.matrix)
obs_coordinates(data::DataMatrix{<:Union{SVD,LowRank}}) = obs_coordinates(data.matrix)
obs_coordinates(data::DataMatrix{<:AbstractMatrix}) = data.matrix


Base.getindex(data::DataMatrix, I::Index, J::Index) = filter_matrix(data, I, J)


_startswith(a::AbstractVector, b::AbstractVector) = length(a)>=length(b) && view(a,1:length(b)) == b

function project_from(data::DataMatrix, base::DataMatrix, from::DataMatrix, args...; kwargs...)
	_startswith(base.models, from.models) || throw(ArgumentError("The \"from\" model history is not consistent with \"base\" model history."))
	models = @view base.models[length(from.models)+1:end]
	project(data, models, args...; kwargs...)
end
function project_from(data::DataMatrix, base::DataMatrix, ::Nothing, args...; kwargs...)
	ind = 0
	if !isempty(data.models)
		last_model = data.models[end]
		ind = findlast(projection_isequal(last_model), base.models)
		if ind === nothing
			throw(ArgumentError("Could not identify starting point for projection, no model matching $last_model found in the base DataMatrix."))
		end
	end
	project(data, (@view base.models[ind+1:end]), args...; kwargs...)
end

function project(data::DataMatrix, models::AbstractVector, args...; verbose=true, kwargs...)
	foldl(models; init=data) do d,m
		verbose && @info "Projecting onto $m"
		project(d,m,args...;verbose, kwargs...)
	end
end
project(data::DataMatrix, base::DataMatrix, args...; from=nothing, kwargs...) = project_from(data, base, from, args...; kwargs...)
function project(data::DataMatrix, model::ProjectionModel, args...; verbose=true, kwargs...)
	model,kwargs = _update_model(model; kwargs...)
	project_impl(data, model, args...; verbose, kwargs...)
end


function _update_annot(old, update::Symbol)
	@assert update in (:copy, :keep)
	update == :copy ? copy(old) : old
end
_update_annot(old, update::Symbol, ::Int) = _update_annot(old, update)

_update_annot(::Any, update::DataFrame, ::Int) = update

_update_annot(::Any, prefix::String, n::Int) = DataFrame(id=string.(prefix, 1:n))

function update_matrix(data::DataMatrix, matrix, model=nothing;
                       var::Union{Symbol,String,DataFrame} = "",
                       obs::Union{Symbol,String,DataFrame} = "",
                       var_id_cols = var isa String ? ["id"] : data.var_id_cols,
                       obs_id_cols = obs isa String ? ["id"] : data.obs_id_cols)
	models = model !== nothing ? vcat(data.models,model) : ProjectionModel[]
	var = _update_annot(data.var, var, size(matrix,1))
	obs = _update_annot(data.obs, obs, size(matrix,2))
	DataMatrix(matrix, var, obs, copy(var_id_cols), copy(obs_id_cols), models)
end




# - show -

function _showoverview(io, data)
	sz = size(data)
	print(io, "DataMatrix (", sz[1], " variables and ", sz[2], " observations)")
end

_showmatrix(io, matrix::T) where T = print(io, T)
_showmatrix(io, matrix::SVD) = print(io, "SVD (", innersize(matrix), " dimensions)")
_showmatrix(io, matrix::LowRank) = print(io, "LowRank (", innersize(matrix), " dimensions)")
_showmatrix(io, matrix::MatrixExpression) = show(io, matrix)

function _printannotation(io, name, show_delim; kwargs...)
	show_delim && print(io, ", ")
	printstyled(io, name; kwargs...)
end
function _showannotations(io, annotations, id_cols, header)
	n = names(annotations)
	print(io, header, ": ")
	actual_id_cols = intersect(n, id_cols)
	bad_id_cols = setdiff(id_cols, actual_id_cols)
	other_cols = setdiff(n, actual_id_cols)

	c = 0
	for name in actual_id_cols
		_printannotation(io, name, c>0; underline=true)
		c += 1
	end
	for name in bad_id_cols
		_printannotation(io, name, c>0; underline=true, reverse=true)
		c += 1
	end
	for name in other_cols
		c>=10 && (print(io, ", ..."); break)
		_printannotation(io, name, c>0)
		c += 1
	end
end
_showvar(io, data) = _showannotations(io, data.var, data.var_id_cols, "Variables")
_showobs(io, data) = _showannotations(io, data.obs, data.obs_id_cols, "Observations")

function _showmodels(io, models)
	print(io, "Models: ")
	isempty(models) && return
	show(io, MIME"text/plain"(), models[end])
	length(models)==1 && return

	print(io, ", ")
	join(IOContext(io, :compact=>true), models[end-1:-1:max(1,end-5+1)], ", ")
	length(models)>5 && print(io, ", ...")
end


function Base.show(io::IO, ::MIME"text/plain", data::DataMatrix)
	_showoverview(io, data)
	println(io)
	print(io, "  ")
	_showmatrix(io, data.matrix)
	println(io)
	print(io, "  ")
	_showvar(io, data)
	println(io)
	print(io, "  ")
	_showobs(io, data)
	if !isempty(data.models)
		println(io)
		print(io, "  ")
		_showmodels(io, data.models)
	end
end
function Base.show(io::IO, data::DataMatrix)
	sz = size(data)
	print(io, sz[1], '×', sz[2], ' ')
	_showmatrix(io, data.matrix)
end
