function logtransform_matrix(::Preprocessing, T, matrix; scale_factor)
	hblock_map(matrix) do x
		create_spec(SCPCore.logtransform_matrix, T, x; scale_factor, __version=v"0.2.0")
	end
end


function logtransform(f::Union{Mat,Var}, T::DataType, data; scale_factor)
	matrix_spec = get_matrix_spec(data)
	create_spec(Preprocess{false}(logtransform_matrix), T, matrix_spec; scale_factor)
end
logtransform(::Var, ::DataType, data; kwargs...) = get_var_spec(data)
logtransform(::Obs, ::DataType, data; kwargs...) = get_obs_spec(data)


function Jobs.logtransform(T::DataType, counts; scale_factor=10_000, kwargs...)
	create_spec(DataMatrixFunction(logtransform), T, counts; scale_factor, kwargs...)
end
Jobs.logtransform(counts; kwargs...) = Jobs.logtransform(Float64, counts; kwargs...)



# ------------------------------------------------------------------------------


# function logcellcounts_impl(X, var_ind)
# 	feature_mask = falses(size(X,1))
# 	feature_mask[var_ind] .= true
# 	SCTransform.logcellcounts(X, feature_mask)
# end
# logcellcounts_spec(X, var_ind) = create_spec(logcellcounts_impl, X, var_ind; __version=v"0.1.0")

function logcellcounts_impl(X, var_ind)
	s = SCPCore.counts_sum(identity, X, var_ind; dims=1)
	log10.(max.(1,s))
end
function logcellcounts_blocked(::Preprocessing, X, var_ind)
	hblock_map(X; wrap=(a,_)->vcat_spec(a)) do x
		create_spec(logcellcounts_impl, x, var_ind; __version=v"0.1.1")
	end
end
logcellcounts_spec(X, var_ind) = create_spec(Preprocess{false}(logcellcounts_blocked), X, var_ind)



function loggenemean_impl(X)
	N = size(X,2)
	obs_ind = 1:size(X,2)
	s = SCPCore.counts_sum(log1p, X, obs_ind; dims=2) # TODO: Avoid passing ind since we want all
	log10.(expm1.(s./N))
end
loggenemean_spec(X) = create_spec(loggenemean_impl, X; __version=v"0.1.0")


function scparams_impl(::Type{Tv}, ::Type{Ti}, matrix; var_ind, log_cell_counts::ROVec, log_gene_mean::ROVec) where {Tv,Ti}
	log_cell_counts = parent(log_cell_counts)
	log_gene_mean = parent(log_gene_mean)

	feature_mask = falses(size(matrix,1))
	feature_mask[var_ind] .= true

	progress = ProgressBar(styled"{blue:  ┌─}")

	df = DataFrame(SCTransform.compute_scparams(Tv, Ti, matrix; log_cell_counts, log_gene_mean, feature_mask, verbose=false, progress); copycols=false)
	table_to_compound_result(df)
end
scparams_impl(matrix::SparseMatrixCSC{Tv,Ti}; kwargs...) where {Tv,Ti} = scparams_impl(Tv, Ti, matrix; kwargs...)
scparams_impl(matrix::Blocks{SparseMatrixCSC{Tv,Ti}}; kwargs...) where {Tv,Ti} = scparams_impl(Tv, Ti, matrix; kwargs...)


create_scparams_impl_spec(matrix; var_ind, log_cell_counts, log_gene_mean) =
	table_from_compound_result(create_spec(scparams_impl, matrix; var_ind=prefetched(var_ind), log_cell_counts, log_gene_mean, __version=v"0.1.2"))


function scparams(action::Action, matrix, var, var_ind; log_cell_counts)
	# The inference is always done for the "eval" case
	log_gene_mean = loggenemean_spec(matrix)
	params = create_scparams_impl_spec(matrix; var_ind, log_cell_counts, log_gene_mean) # DataFrame, but without IDs

	if action isa Eval
		return params
	else#if actions is Projection
		# We need to remap IDs
		var_ids = id_column_spec(var)
		var_ids2 = action(var_ids)

		param_ids = table_getindex_spec(var_ids, var_ind) # The IDs represented in the params table
		var_ids_proj = intersect_ids_spec(param_ids, var_ids2)
		var_ind_proj = indexin_spec(var_ids_proj, param_ids; not_found=:error)
		return table_getindex_spec(params, prefetched(var_ind_proj))
	end
end
create_scparams_spec(matrix, var, var_ind; log_cell_counts) =
	create_spec(Projectable(scparams), matrix, var, var_ind; log_cell_counts)



sctransformsparse_a_spec(T, matrix, params, var_ind, log_cell_counts; kwargs...) =
	create_spec(SCPCore.sctransformsparse_a, T, matrix, params, var_ind, log_cell_counts; kwargs..., __version=v"0.1.0")


function sctransform_matrix_a_impl(::Preprocessing, T, matrix, params, var_ind, log_cell_counts; kwargs...)
	# TODO: simplify handling of hblocked matrix with matching log_cell_counts
	if is_hblock(matrix)
		n = length(matrix.args[1])
		@assert log_cell_counts.f == vcat_impl
		@assert length(log_cell_counts.args[1]) == n # These should match because we must have the same samples

		samples = Vector{Spec}(undef, n)
		for i in 1:n
			X = matrix.args[1][i]
			lcc = log_cell_counts.args[1][i]
			# samples[i] = create_spec(SCPCore.sctransformsparse_a, T, X, params, var_ind, lcc; kwargs..., __version=v"0.1.0")
			samples[i] = sctransformsparse_a_spec(T, X, params, var_ind, lcc; kwargs...)
		end
		hblock_spec(samples, _get_kwarg(matrix, :ranges))
	else
		# create_spec(SCPCore.sctransformsparse_a, T, matrix, params, var_ind, log_cell_counts; kwargs..., __version=v"0.1.0")
		sctransformsparse_a_spec(T, matrix, params, var_ind, log_cell_counts; kwargs...)
	end
end



# matrix[var_ind,:] must match params exactly
function sctransform_matrix_pr(action::Action, T, matrix, params, log_cell_counts; var_ind, nobs=nothing, clip=nothing, rtol=1e-3, atol=0.0)
	@assert nobs !== nothing || clip !== nothing "Must specify either nobs or clip"
	clip = @something clip sqrt(nobs/30)
	# create_spec(SCPCore.sctransform_matrix, T, action(matrix), action(params), action(var_ind), action(log_cell_counts); clip, rtol, atol, __version=v"0.1.0")

	matrix = action(matrix)
	params = action(params)
	var_ind = action(var_ind)
	log_cell_counts = action(log_cell_counts)

	a_spec = create_spec(Preprocess{false}(sctransform_matrix_a_impl), T, matrix, params, var_ind, log_cell_counts; clip)

	# Get row and col ranges for the blocked input matrix and pass them onto b_spec so that B₁ and B₃ can be blocked in the same way.
	row_ranges = get_row_ranges_spec(a_spec)
	col_ranges = get_col_ranges_spec(a_spec)

	# b_spec = create_spec(SCPCore.sctransformsparse_b, params, log_cell_counts; rtol, atol, __version=v"0.1.2")
	b_spec = create_spec(SCPCore.sctransformsparse_b, params, log_cell_counts; row_ranges, col_ranges, rtol, atol, __version=v"0.1.2")
	matrix_sum_impl_spec(:A=>a_spec, b_spec)
end

sctransform_matrix_spec(T, matrix, params, log_cell_counts; var_ind, kwargs...) =
	create_spec(Projectable(sctransform_matrix_pr), T, matrix, params, log_cell_counts; var_ind, kwargs...)



function sctransform(f::Union{Mat,Var}, ::Type{T}, counts; var_filter=:, min_cells=5, annotate=false, kwargs...) where T
	matrix_spec = get_matrix_spec(counts)
	var_spec = get_var_spec(counts)

	var_ind_logcellcounts = prefetched(create_find_matching_ind_spec(var_filter, var_spec; project_ids=:intersect))
	log_cell_counts = logcellcounts_spec(matrix_spec, var_ind_logcellcounts)

	# min_cells
	nnz_cells = cached(counts_sum_impl_spec(!iszero, matrix_spec, :; dims=2)) # returns vector
	var_nnz_cells = add_column_spec(id_column_spec(var_spec), "nnzCells", nnz_cells)
	var_ind_min_cells = create_find_matching_ind_spec("nnzCells"=>>=(min_cells), var_nnz_cells; project_ids=:yes)

	var_ind = prefetched(intersect_ind_spec(var_ind_logcellcounts, var_ind_min_cells))
	params_spec = create_scparams_spec(matrix_spec, var_spec, var_ind; log_cell_counts)

	if f isa Var
		var_out = table_getindex_spec(var_spec, var_ind)
		if annotate
			var_out = table_hcat_spec(var_out, params_spec)
		end
		return var_out
	else # if f isa Mat
		nobs = fetched(nobs_spec(counts)) # fetch since we need the value now and the value should **not** be affected by projecion
		return sctransform_matrix_spec(T, matrix_spec, params_spec, log_cell_counts; var_ind, nobs, kwargs...)
	end
end
sctransform(::Obs, ::DataType, counts; kwargs...) = get_obs_spec(counts)



function Jobs.sctransform(T::DataType, counts; kwargs...)
	create_spec(DataMatrixFunction(sctransform), T, counts; kwargs...)
end
Jobs.sctransform(counts; kwargs...) = Jobs.sctransform(Float64, counts; kwargs...)
