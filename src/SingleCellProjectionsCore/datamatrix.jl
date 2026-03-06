function _get_nonunique(table, col)
	bad_ind = findfirst(nonunique(table, col))
	bad_ind !== nothing ? table[bad_ind,col] : nothing
end

function _validateunique(table, col, report, suffix)
	report == :ignore && return
	@assert report in (:warn, :error)
	bad_id = _get_nonunique(table,col)
	bad_id === nothing && return
	msg = string("ID \"", bad_id, "\" is not unique.", suffix)
	report == :error && error(msg)
	report == :warn && @warn(msg)
	return
end

validateunique_var(table, col; report) = _validateunique(table,col,report," Use duplicate_var=x, where x is :error, :warn or :ignore to control behavior.")
validateunique_obs(table, col; report) = _validateunique(table,col,report," Use duplicate_obs=x, where x is :error, :warn or :ignore to control behavior.")


"""
	struct DataMatrix{T}

A `DataMatrix` represents a matrix together with annotations for variables and observations.

Fields:
* `matrix::T` - The matrix.
* `var::DataFrame` - Variable annotations.
* `obs::DataFrame` - Observation annotations.

The first column of the `var` and `obs` tables should contain unique IDs.

Main constructor:

	DataMatrix(matrix, var::DataFrame, obs::DataFrame; kwargs...)

Create a `DataMatrix`, from the given `matrix`, `var` and `obs`.

The first column of `var`/`obs` are used as IDs.

Kwargs:
* `duplicate_var` - Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate var IDs are found.
* `duplicate_obs` - Set to `:ignore`, `:warn` or `:error` to decide what happens if duplicate obs IDs are found.
"""
struct DataMatrix{T}
	matrix::T
	var::DataFrame
	obs::DataFrame

	function DataMatrix(matrix::T, var::DataFrame, obs::DataFrame; duplicate_var=:warn, duplicate_obs=:warn) where T
		P,N = size(matrix)
		p = size(var,1)
		n = size(obs,1)
		if P != p || N != n
			throw(DimensionMismatch(string("Matrix has dimensions (",P,',',N,"), but there are ", p, " variable annotations and ", n, " observation annotations.")))
		end

		validateunique_var(var, 1; report=duplicate_var)
		validateunique_obs(obs, 1; report=duplicate_obs)
		new{T}(matrix, var, obs)
	end
end




"""
	DataMatrix()

Create an empty DataMatrix{Matrix{Float64}}.
"""
DataMatrix() = DataMatrix(zeros(0,0),DataFrame(id=String[]),DataFrame(id=String[]))

Base.:(==)(a::DataMatrix, b::DataMatrix) = false
function Base.:(==)(a::DataMatrix{T}, b::DataMatrix{T}) where T
	all(i->getfield(a,i)==getfield(b,i), 1:nfields(a))
end


Base.size(data::DataMatrix) = size(data.matrix)
Base.size(data::DataMatrix, dim::Integer) = size(data.matrix, dim)

Base.axes(data::DataMatrix, d::Integer) = axes(data)[d] # needed to make end work in getindex





get_matrix(data::DataMatrix) = data.matrix
get_var(data::DataMatrix) = data.var
get_obs(data::DataMatrix) = data.obs

get_var_ids(data::DataMatrix) = data.var[!,1:1]
get_obs_ids(data::DataMatrix) = data.obs[!,1:1]





# - show -

function _showoverview(io, data)
	sz = size(data)
	print(io, "DataMatrix (", sz[1], " variables and ", sz[2], " observations)")
end

_showmatrix(io, matrix::T) where T = print(io, T)
_showmatrix(io, matrix::SVD) = print(io, "SVD (", innersize(matrix), " dimensions)")
# _showmatrix(io, matrix::LowRank) = print(io, "LowRank (", innersize(matrix), " dimensions)")
_showmatrix(io, matrix::MatrixExpression) = show(io, matrix)

function _printannotation(io, name, show_delim; kwargs...)
	show_delim && print(io, ", ")
	printstyled(io, name; kwargs...)
end
function _showannotations(io, annotations, header)
	print(io, header, ": ")
	for (c,name) in enumerate(names(annotations))
		c>=10 && (print(io, ", ..."); break)
		_printannotation(io, name, c>1; underline=c==1, reverse=c==1 && !allunique(annotations[!,name])) # reverse if ID column doesn't have unique values
	end
end
_showvar(io, data) = _showannotations(io, data.var, "Variables")
_showobs(io, data) = _showannotations(io, data.obs, "Observations")

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
end
function Base.show(io::IO, data::DataMatrix)
	sz = size(data)
	print(io, sz[1], '×', sz[2], ' ')
	_showmatrix(io, data.matrix)
end
