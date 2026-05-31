col_sum_squared_spec(X) =
	cached(create_spec(SCPCore.col_sum_squared, X; __version=v"0.1.1"))
row_sum_squared_spec(X) =
	cached(create_spec(SCPCore.row_sum_squared, X; __version=v"0.1.1"))


sum_squared_to_var_spec(s2, n) =
	create_spec(SCPCore.sum_squared_to_var, s2, n; __version=v"0.1.0")


function compute_variance(action::Action, X; col="variance", project=:no)
	@assert project in (:no, :yes)
	matrix = project == :yes ? action(get_matrix_spec(X)) : get_matrix_spec(X)
	s2 = row_sum_squared_spec(matrix)
	n = fetched(nobs_spec(X))
	values = cached(sum_squared_to_var_spec(s2, n))
	table_hcat_spec(id_column_spec(get_var_spec(X)),
	                create_table_spec(col => values))
end

variance_spec(X; kwargs...) = create_spec(Projectable(compute_variance), X; kwargs...)
function Jobs.variance(X; kwargs...)
	variance_spec(X; kwargs...)
end


compute_std(::Preprocessing, X; col="std", project=:no) =
	transform_annotation_spec(sqrt, variance_spec(X; project); new_name=col)

std_spec(X; kwargs...) = create_spec(Preprocess(compute_std), X; kwargs...)
function Jobs.std(X; kwargs...)
	std_spec(X; kwargs...)
end
