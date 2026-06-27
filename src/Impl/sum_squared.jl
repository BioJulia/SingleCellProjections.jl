col_sum_squared_job(X) =
	cached(create_job(SCPCore.col_sum_squared, X; __version=v"0.1.1"))
row_sum_squared_job(X) =
	cached(create_job(SCPCore.row_sum_squared, X; __version=v"0.1.1"))


sum_squared_to_var_job(s2, n) =
	create_job(SCPCore.sum_squared_to_var, s2, n; __version=v"0.1.0")


function compute_variance(action::Action, X; assume_centered::Bool, col="variance", project=:no)
	@assert project in (:no, :yes)
	# variance/std/relative_std are computed assuming a mean of zero, so the caller must
	# confirm the data is mean-centered. `false` is not supported.
	assume_centered || throw(ArgumentError("assume_centered must be `true`: Consider using `normalize_matrix` to compute variance/std/relative_std."))
	project == :yes && (X = action(X))
	matrix = get_matrix_job(X)
	s2 = row_sum_squared_job(matrix)
	n = fetched(nobs_job(X))
	values = cached(sum_squared_to_var_job(s2, n))
	table_hcat_job(id_column_job(get_var_job(X)), create_table_job(col => values))
end

variance_job(X; kwargs...) = create_job(Projectable(compute_variance), X; kwargs...)


function compute_std(::Preprocessing, X; assume_centered::Bool, col="std", project=:no)
	transform_annotation_job(sqrt, variance_job(X; assume_centered, project); new_name=col)
end

std_job(X; kwargs...) = create_job(Preprocess(compute_std), X; kwargs...)


function compute_relative_std(::Preprocessing, X; assume_centered::Bool, col="relative_std", project=:no)
	std_table = std_job(X; assume_centered, project)
	max_std = prefetched(apply_job(maximum, value_column_data_job(std_table)))
	transform_annotation_job(Base.Fix2(/, max_std), std_table; new_name=col)
end

relative_std_job(X; kwargs...) = create_job(Preprocess(compute_relative_std), X; kwargs...)
