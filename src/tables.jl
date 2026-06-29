"""
    SCP.create_table(col1 => values1, col2 => values2, ...) -> Job

Create a new table `Job` from column name/value pairs.
"""
create_table(args...) = create_job(Impl.create_table, args...; __version=v"0.1.0")

"""
    SCP.get_colnames(table; kwargs...) -> Job

Return the column names of `table`.

See also [`get_id_colname`](@ref), [`get_value_colname`](@ref).
"""
get_colnames(table, args...; kwargs...) = create_job(Preprocess(Impl.get_colnames), table, args...; kwargs...)

"""
    SCP.get_id_colname(table) -> Job

Return the name of the first (ID) column of `table`.

See also [`get_colnames`](@ref), [`get_value_colname`](@ref).
"""
get_id_colname(table) = create_job(Preprocess(Impl.get_colnames), table, 1)

"""
    SCP.get_value_colname(table) -> Job

Return the name of the second (value) column of `table`. Requires the table to have
exactly two columns.

See also [`get_colnames`](@ref), [`get_id_colname`](@ref).
"""
get_value_colname(table) = create_job(Preprocess(Impl.get_colnames), table, 2; require_n_cols=2)

"""
    SCP.get_columns(table, colnames...) -> Job

Select specific columns from `table` by name or index.

See also [`id_column`](@ref), [`value_column`](@ref).
"""
get_columns(table, colname1, colnames...; kwargs...) = create_job(Preprocess(Impl.get_columns), table, colname1, colnames...; kwargs...)

"""
    SCP.id_column(table) -> Job

Extract the first (ID) column of `table` as a single-column table.

See also [`value_column`](@ref), [`id_column_data`](@ref).
"""
id_column(table) = create_job(Preprocess(Impl.id_column), table)

"""
    SCP.value_column(table) -> Job

Extract the second (value) column of `table` as a single-column table.

See also [`id_column`](@ref), [`value_column_data`](@ref).
"""
value_column(table) = create_job(Preprocess(Impl.value_column), table)

"""
    SCP.annotation(table, colname) -> Job

Extract the ID column and the column named `colname` from `table`, returning a two-column
table. Useful for passing annotations to filtering or covariate specification.
"""
annotation(table, colname) = create_job(Preprocess(Impl.annotation), table, colname)

"""
    SCP.column_data(table, col; kwargs...) -> Job

Return the values of column `col` from `table` as a vector.

See also [`id_column_data`](@ref), [`value_column_data`](@ref).
"""
column_data(table, col; kwargs...) = create_job(Preprocess(Impl.column_data), table, col; kwargs...)

"""
    SCP.id_column_data(table) -> Job

Return the vector of IDs (first column) from `table`.

See also [`column_data`](@ref), [`value_column_data`](@ref).
"""
id_column_data(table) = create_job(Preprocess(Impl.id_column_data), table)

"""
    SCP.value_column_data(table) -> Job

Return the values (second column) from `table` as a vector. Requires the table to have
exactly two columns.

See also [`column_data`](@ref), [`id_column_data`](@ref).
"""
value_column_data(table) = create_job(Preprocess(Impl.value_column_data), table)

"""
    SCP.table_nrow(table) -> Job

Return the number of rows in `table`.

See also [`table_ncol`](@ref).
"""
table_nrow(table) = create_job(Preprocess(Impl.table_nrow), table)

"""
    SCP.table_ncol(table) -> Job

Return the number of columns in `table`.

See also [`table_nrow`](@ref).
"""
table_ncol(table) = create_job(Preprocess(Impl.table_ncol), table)

"""
    SCP.add_column(table, name, column) -> Job

Add a column named `name` with values `column` to `table`.
The length of `column` must match the number of rows in `table`.

See also [`table_hcat`](@ref), [`add_var_column`](@ref), [`add_obs_column`](@ref).
"""
add_column(table, name, column) = create_job(Preprocess(Impl.add_column), table, name, column)

"""
    SCP.table_hcat(a, tables...) -> Job

Horizontally concatenate tables. All tables must have the same number of rows and
matching row order.

See also [`table_leftjoin`](@ref), [`add_column`](@ref).
"""
table_hcat(a, args...) = create_job(Preprocess(Impl.table_hcat), a, args...)

"""
    SCP.table_leftjoin(a, b) -> Job

Left-join table `b` onto table `a` by their ID columns.

See also [`table_hcat`](@ref), [`annotate_var`](@ref), [`annotate_obs`](@ref).
"""
table_leftjoin(a, b) = create_job(Preprocess(Impl.table_leftjoin), a, b)

"""
    SCP.transform_annotation(f, table; kwargs...) -> Job

Apply function `f` element-wise to the value column of `table`, returning a new table
with transformed values. The table must have exactly two columns (ID and value).
Use `new_name` to rename the value column.

(TODO: Example.)
"""
transform_annotation(f, table; kwargs...) = create_job(Preprocess(Impl.transform_annotation), f, table; kwargs...)
