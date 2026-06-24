module MuonExt

using ReproducibleJobs
using ReproducibleJobs: create_job, cached, prefetched, ChecksummedFilePath
using SingleCellProjections
using SingleCellProjections: DataMatrixFunction, Mat, Var, Obs, table_to_compound_result, table_from_compound_result, checksummedfilepath_job, prefixed_ids_job, compute_size_job
using .SingleCellProjections.SingleCellProjectionsCore
using DataFrames
using SparseArrays: SparseMatrixCSC
import LinearAlgebra

using HDF5: h5open
using Muon: AnnData

# We use Muon.jl to read .h5ad files to avoid maintaining a reader ourselves.
# Unfortunately, Muon's reader is only partially lazy, so this is a bit wasteful for our use case.
# Still, it gets the job done, and it will probably work fine for most use cases.


function convert_matrix(::Type{T}, X) where T
	if !(eltype(X) <: T)
		convert.(T, X) # handles both sparse and dense cases, gets rid of transposes
	elseif X isa LinearAlgebra.Adjoint
		copy(X) # materialize transpose
	else
		X
	end
end

function _transpose(X::PermutedDimsArray)
	Xt = parent(X)
	@assert PermutedDimsArray(Xt, (2,1)) === X
	Xt
end
_transpose(X) = X'

function _read_h5ad(f, filepath)
	@assert filepath isa ChecksummedFilePath
	# This workaround is needed because Muon by default opens backed files in "r+" which can change the mtime.
	h5open(string(filepath), "r") do fid
		ann = AnnData(fid, true, false)
		f(ann)
	end
end


function load_h5ad_var_impl(filepath)
	_read_h5ad(filepath) do ann
		df = insertcols(ann.var, 1, :id => collect(ann.var_names); makeunique=true)
		table_to_compound_result(df)
	end
end
load_h5ad_var_job(filepath) = create_job(load_h5ad_var_impl, filepath; __version=v"0.1.0")

function load_h5ad_obs_impl(filepath)
	_read_h5ad(filepath) do ann
		df = insertcols(ann.obs, 1, :cell_id => collect(ann.obs_names); makeunique=true)
		table_to_compound_result(df)
	end
end
load_h5ad_obs_job(filepath) = create_job(load_h5ad_obs_impl, filepath; __version=v"0.1.0")

function load_h5ad_matrix_impl(filepath; T, layer=nothing, obsm=nothing, obsp=nothing, varm=nothing, varp=nothing, row_block_size=1024, col_block_size=1024)
	_read_h5ad(filepath) do ann
		# X and layers are lazy (backed) and need read(), obsm/obsp/varm/varp are eagerly loaded
		X = if layer !== nothing
			read(ann.layers[layer])
		elseif obsm !== nothing
			ann.obsm[obsm]
		elseif obsp !== nothing
			ann.obsp[obsp]
		elseif varm !== nothing
			ann.varm[varm]
		elseif varp !== nothing
			ann.varp[varp]
		else
			read(ann.X)
		end
		if varm === nothing && varp === nothing
			X = _transpose(X)
		end
		X = convert_matrix(T, X)
		if X isa SparseMatrixCSC
			X = SingleCellProjectionsCore.blockify(X; row_block_size, col_block_size)
		end
		X
	end
end
load_h5ad_matrix_job(filepath; kwargs...) = create_job(load_h5ad_matrix_impl, filepath; T=Any, kwargs..., __version=v"0.1.0")


load_h5ad(::Mat, filepath; kwargs...) = load_h5ad_matrix_job(filepath; kwargs...)

function load_h5ad(::Var, filepath; obsm=nothing, obsp=nothing, kwargs...)
	if obsm !== nothing
		mat_job = load_h5ad(Mat(), filepath; obsm, kwargs...)
		prefixed_ids_job("id", "Dim", prefetched(compute_size_job(mat_job, 1)))
	elseif obsp !== nothing
		table_from_compound_result(cached(load_h5ad_obs_job(filepath)))
	else
		table_from_compound_result(cached(load_h5ad_var_job(filepath)))
	end
end

function load_h5ad(::Obs, filepath; varm=nothing, varp=nothing, kwargs...)
	if varm !== nothing
		mat_job = load_h5ad(Mat(), filepath; varm, kwargs...)
		prefixed_ids_job("id", "Dim", prefetched(compute_size_job(mat_job, 2)))
	elseif varp !== nothing
		table_from_compound_result(cached(load_h5ad_var_job(filepath)))
	else
		table_from_compound_result(cached(load_h5ad_obs_job(filepath)))
	end
end

function Jobs.load_h5ad(filepath; kwargs...)
	if count(key->haskey(kwargs,key), (:layer, :obsm, :obsp, :varm, :varp)) > 1
		throw(ArgumentError("At most one of layer, obsm, obsp, varm, varp can be specified."))
	end

	filepath_job = checksummedfilepath_job(filepath)
	create_job(DataMatrixFunction(load_h5ad), filepath_job; kwargs...)
end
Jobs.load_h5ad(::Type{T}, filepath; kwargs...) where T = Jobs.load_h5ad(filepath; T, kwargs...)


end
