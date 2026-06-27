function logtransform_matrix(::Preprocessing, T, matrix; scale_factor)
	hblock_map(matrix) do x
		create_job(SCPCore.logtransform_matrix, T, x; scale_factor, __version=v"0.2.0")
	end
end


function logtransform(f::Union{Mat,Var}, T::DataType, data; scale_factor)
	matrix_job = get_matrix_job(data)
	create_job(Preprocess{false}(logtransform_matrix), T, matrix_job; scale_factor)
end
logtransform(::Var, ::DataType, data; kwargs...) = get_var_job(data)
logtransform(::Obs, ::DataType, data; kwargs...) = get_obs_job(data)



# ------------------------------------------------------------------------------


# function logcellcounts_impl(X, var_ind)
# 	feature_mask = falses(size(X,1))
# 	feature_mask[var_ind] .= true
# 	SCTransform.logcellcounts(X, feature_mask)
# end
# logcellcounts_job(X, var_ind) = create_job(logcellcounts_impl, X, var_ind; __version=v"0.1.0")

function logcellcounts_impl(X, var_ind)
	s = SCPCore.counts_sum(identity, X, var_ind; dims=1)
	log10.(max.(1,s))
end
function logcellcounts_blocked(::Preprocessing, X, var_ind)
	hblock_map(X; wrap=(a,_)->vcat_job(a)) do x
		create_job(logcellcounts_impl, x, var_ind; __version=v"0.1.1")
	end
end
logcellcounts_job(X, var_ind) = create_job(Preprocess{false}(logcellcounts_blocked), X, var_ind)



function loggenemean_impl(X)
	N = size(X,2)
	obs_ind = 1:size(X,2)
	s = SCPCore.counts_sum(log1p, X, obs_ind; dims=2) # TODO: Avoid passing ind since we want all
	log10.(expm1.(s./N))
end
loggenemean_job(X) = create_job(loggenemean_impl, X; __version=v"0.1.0")


function scparams_impl(::Type{Tv}, ::Type{Ti}, matrix; var_ind, log_cell_counts::ROVec, log_gene_mean::ROVec) where {Tv,Ti}
	log_cell_counts = parent(log_cell_counts)
	log_gene_mean = parent(log_gene_mean)

	feature_mask = falses(size(matrix,1))
	feature_mask[var_ind] .= true

	progress = ProgressBar(styled"{blue:  ┌─}")

	df = DataFrame(SCTransform.compute_scparams(Tv, Ti, matrix; log_cell_counts, log_gene_mean, feature_mask, verbose=false, progress, tick=throw_if_cancelled); copycols=false)
	table_to_compound_result(df)
end
scparams_impl(matrix::SparseMatrixCSC{Tv,Ti}; kwargs...) where {Tv,Ti} = scparams_impl(Tv, Ti, matrix; kwargs...)
scparams_impl(matrix::Blocks{SparseMatrixCSC{Tv,Ti}}; kwargs...) where {Tv,Ti} = scparams_impl(Tv, Ti, matrix; kwargs...)


create_scparams_impl_job(matrix; var_ind, log_cell_counts, log_gene_mean) =
	table_from_compound_result(create_job(scparams_impl, matrix; var_ind=prefetched(var_ind), log_cell_counts, log_gene_mean, __version=v"0.1.2"))


function scparams(action::Action, matrix, var, var_ind; log_cell_counts)
	# The inference is always done for the "eval" case
	log_gene_mean = loggenemean_job(matrix)
	params = create_scparams_impl_job(matrix; var_ind, log_cell_counts, log_gene_mean) # DataFrame, but without IDs

	if action isa Eval
		return params
	else#if actions is Projection
		# We need to remap IDs
		var_ids = id_column_job(var)
		var_ids2 = action(var_ids)

		param_ids = table_getindex_job(var_ids, var_ind) # The IDs represented in the params table
		var_ids_proj = intersect_ids_job(param_ids, var_ids2)
		var_ind_proj = indexin_job(var_ids_proj, param_ids; not_found=:error)
		return table_getindex_job(params, prefetched(var_ind_proj))
	end
end
create_scparams_job(matrix, var, var_ind; log_cell_counts) =
	create_job(Projectable(scparams), matrix, var, var_ind; log_cell_counts)



sctransformsparse_a_job(T, matrix, params, var_ind, log_cell_counts; kwargs...) =
	create_job(SCPCore.sctransformsparse_a, T, matrix, params, var_ind, log_cell_counts; kwargs..., __version=v"0.1.0")


function sctransform_matrix_a_impl(::Preprocessing, T, matrix, params, var_ind, log_cell_counts; kwargs...)
	# TODO: simplify handling of hblocked matrix with matching log_cell_counts
	if is_hblock(matrix)
		n = length(matrix.args[1])
		@assert log_cell_counts.f == vcat_impl
		@assert length(log_cell_counts.args[1]) == n # These should match because we must have the same samples

		samples = Vector{SpecRef}(undef, n)
		for i in 1:n
			X = matrix.args[1][i]
			lcc = log_cell_counts.args[1][i]
			# samples[i] = create_job(SCPCore.sctransformsparse_a, T, X, params, var_ind, lcc; kwargs..., __version=v"0.1.0")
			samples[i] = sctransformsparse_a_job(T, X, params, var_ind, lcc; kwargs...)
		end
		hblock_job(samples, _get_kwarg(matrix, :ranges))
	else
		# create_job(SCPCore.sctransformsparse_a, T, matrix, params, var_ind, log_cell_counts; kwargs..., __version=v"0.1.0")
		sctransformsparse_a_job(T, matrix, params, var_ind, log_cell_counts; kwargs...)
	end
end



# matrix[var_ind,:] must match params exactly
function sctransform_matrix_pr(action::Action, T, matrix, params, log_cell_counts; var_ind, nobs=nothing, clip=nothing, rtol=1e-3, atol=0.0)
	@assert nobs !== nothing || clip !== nothing "Must specify either nobs or clip"
	clip = @something clip sqrt(nobs/30)
	# create_job(SCPCore.sctransform_matrix, T, action(matrix), action(params), action(var_ind), action(log_cell_counts); clip, rtol, atol, __version=v"0.1.0")

	matrix = action(matrix)
	params = action(params)
	var_ind = action(var_ind)
	log_cell_counts = action(log_cell_counts)

	a_job = create_job(Preprocess{false}(sctransform_matrix_a_impl), T, matrix, params, var_ind, log_cell_counts; clip)

	# Get row and col ranges for the blocked input matrix and pass them onto b_job so that B₁ and B₃ can be blocked in the same way.
	row_ranges = get_row_ranges_job(a_job)
	col_ranges = get_col_ranges_job(a_job)

	# b_job = create_job(SCPCore.sctransformsparse_b, params, log_cell_counts; rtol, atol, __version=v"0.1.2")
	b_job = create_job(SCPCore.sctransformsparse_b, params, log_cell_counts; row_ranges, col_ranges, rtol, atol, __version=v"0.1.2")
	matrix_sum_impl_job(:A=>a_job, b_job)
end

sctransform_matrix_job(T, matrix, params, log_cell_counts; var_ind, kwargs...) =
	create_job(Projectable(sctransform_matrix_pr), T, matrix, params, log_cell_counts; var_ind, kwargs...)



function sctransform(f::Union{Mat,Var}, ::Type{T}, counts; var_filter=:, min_cells=5, annotate=false, kwargs...) where T
	matrix_job = get_matrix_job(counts)
	var_job = get_var_job(counts)

	var_ind_logcellcounts = prefetched(create_find_matching_ind_job(var_filter, var_job; project_ids=:intersect))
	log_cell_counts = logcellcounts_job(matrix_job, var_ind_logcellcounts)

	# min_cells
	nnz_cells = cached(counts_sum_impl_job(!iszero, matrix_job, :; dims=2)) # returns vector
	var_nnz_cells = add_column_job(id_column_job(var_job), "nnzCells", nnz_cells)
	var_ind_min_cells = create_find_matching_ind_job("nnzCells"=>>=(min_cells), var_nnz_cells; project_ids=:yes)

	var_ind = prefetched(intersect_ind_job(var_ind_logcellcounts, var_ind_min_cells))
	params_job = create_scparams_job(matrix_job, var_job, var_ind; log_cell_counts)

	if f isa Var
		var_out = table_getindex_job(var_job, var_ind)
		if annotate
			var_out = table_hcat_job(var_out, params_job)
		end
		return var_out
	else # if f isa Mat
		nobs = fetched(nobs_job(counts)) # fetch since we need the value now and the value should **not** be affected by projecion
		return sctransform_matrix_job(T, matrix_job, params_job, log_cell_counts; var_ind, nobs, kwargs...)
	end
end
sctransform(::Obs, ::DataType, counts; kwargs...) = get_obs_job(counts)



