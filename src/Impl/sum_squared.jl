col_sum_squared_impl_job(X) =
	create_job(SCPCore.col_sum_squared, X; __version=v"1.0.0")
row_sum_squared_impl_job(X) =
	create_job(SCPCore.row_sum_squared, X; __version=v"1.0.0")


# Block-aware col_sum_squared (per-obs result): compute col_sum_squared per block and vcat. dims=1
# column reduction with no mask, so no index splitting. The per-block results are cached (in the wrap,
# for recurring-sample dedup); the combine is uncached and the non-block fallback returns the uncached
# leaf, so the caller decides whether to cache the combined result (matches counts_sum).
function col_sum_squared(::Preprocessing, X)
	hblock_map(X; wrap=(a,_)->vcat_job(cached.(a))) do x
		col_sum_squared_impl_job(x)
	end
end
col_sum_squared_job(X) = create_job(Preprocess{false}(col_sum_squared), X)


# Block-aware row_sum_squared (per-var result): each column block yields a partial per-var sum of
# squares, combined element-wise with `sum`. dims=2 row reduction with no mask, so (unlike counts_sum
# dims=2) no obs index splitting. Caching as for col_sum_squared above.
function row_sum_squared(::Preprocessing, X)
	hblock_map(X; wrap=(a,_)->apply_job(sum, cached.(a))) do x
		row_sum_squared_impl_job(x)
	end
end
row_sum_squared_job(X) = create_job(Preprocess{false}(row_sum_squared), X)


sum_squared_to_var_job(s2, n) =
	create_job(SCPCore.sum_squared_to_var, s2, n; __version=v"1.0.0")


function compute_variance(action::Action, X; assume_centered::Bool, col="variance", project=:no)
	@assert project in (:no, :yes)
	# variance/std/relative_std are computed assuming a mean of zero, so the caller must
	# confirm the data is mean-centered. `false` is not supported.
	assume_centered || throw(ArgumentError("assume_centered must be `true`: Consider using `normalize_matrix` to compute variance/std/relative_std."))
	project == :yes && (X = action(X))
	matrix = SCP.get_matrix(X)
	s2 = row_sum_squared_job(matrix)
	n = fetched(SCP.nobs(X))
	values = cached(sum_squared_to_var_job(s2, n))
	SCP.table_hcat(SCP.id_column(SCP.get_var(X)), SCP.create_table(col => values))
end


function compute_std(::Preprocessing, X; assume_centered::Bool, col="std", project=:no)
	SCP.transform_annotation(sqrt, SCP.variance(X; assume_centered, project); new_name=col)
end


function compute_relative_std(::Preprocessing, X; assume_centered::Bool, col="relative_std", project=:no)
	std_table = SCP.std(X; assume_centered, project)
	max_std = prefetched(apply_job(maximum, SCP.value_column_data(std_table)))
	SCP.transform_annotation(Base.Fix2(/, max_std), std_table; new_name=col)
end
