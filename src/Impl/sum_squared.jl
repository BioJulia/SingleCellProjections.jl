col_sum_squared_job(X) =
	cached(create_job(SCPCore.col_sum_squared, X; __version=v"1.0.0"))
row_sum_squared_job(X) =
	cached(create_job(SCPCore.row_sum_squared, X; __version=v"1.0.0"))


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
