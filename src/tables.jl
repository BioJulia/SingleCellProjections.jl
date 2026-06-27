"""
    SCP.create_table(col1 => values1, col2 => values2, ...) -> Job

Create a new table `Job` from column name/value pairs.
"""
create_table(args...) = Impl.create_table_job(args...)

"""
    SCP.get_colnames(table; kwargs...) -> Job

Return the column names of `table`.

See also [`get_id_colname`](@ref), [`get_value_colname`](@ref).
"""
get_colnames(table, args...; kwargs...) = Impl.get_colnames_job(table, args...; kwargs...)

"""
    SCP.get_id_colname(table) -> Job

Return the name of the first (ID) column of `table`.

See also [`get_colnames`](@ref), [`get_value_colname`](@ref).
"""
get_id_colname(table) = Impl.get_id_colname_job(table)

"""
    SCP.get_value_colname(table) -> Job

Return the name of the second (value) column of `table`. Requires the table to have
exactly two columns.

See also [`get_colnames`](@ref), [`get_id_colname`](@ref).
"""
get_value_colname(table) = Impl.get_value_colname_job(table)

"""
    SCP.get_columns(table, colnames...) -> Job

Select specific columns from `table` by name or index.

See also [`id_column`](@ref), [`value_column`](@ref).
"""
get_columns(table, colname1, colnames...; kwargs...) = Impl.get_columns_job(table, colname1, colnames...; kwargs...)

"""
    SCP.id_column(table) -> Job

Extract the first (ID) column of `table` as a single-column table.

See also [`value_column`](@ref), [`id_column_data`](@ref).
"""
id_column(table) = Impl.id_column_job(table)

"""
    SCP.value_column(table) -> Job

Extract the second (value) column of `table` as a single-column table.

See also [`id_column`](@ref), [`value_column_data`](@ref).
"""
value_column(table) = Impl.value_column_job(table)

"""
    SCP.annotation(table, colname) -> Job

Extract the ID column and the column named `colname` from `table`, returning a two-column
table. Useful for passing annotations to filtering or covariate specification.
"""
annotation(table, colname) = Impl.annotation_job(table, colname)

"""
    SCP.column_data(table, col; kwargs...) -> Job

Return the values of column `col` from `table` as a vector.

See also [`id_column_data`](@ref), [`value_column_data`](@ref).
"""
column_data(table, col; kwargs...) = Impl.column_data_job(table, col; kwargs...)

"""
    SCP.id_column_data(table) -> Job

Return the vector of IDs (first column) from `table`.

See also [`column_data`](@ref), [`value_column_data`](@ref).
"""
id_column_data(table) = Impl.id_column_data_job(table)

"""
    SCP.value_column_data(table) -> Job

Return the values (second column) from `table` as a vector. Requires the table to have
exactly two columns.

See also [`column_data`](@ref), [`id_column_data`](@ref).
"""
value_column_data(table) = Impl.value_column_data_job(table)

"""
    SCP.table_nrow(table) -> Job

Return the number of rows in `table`.

See also [`table_ncol`](@ref).
"""
table_nrow(table) = Impl.table_nrow_job(table)

"""
    SCP.table_ncol(table) -> Job

Return the number of columns in `table`.

See also [`table_nrow`](@ref).
"""
table_ncol(table) = Impl.table_ncol_job(table)

"""
    SCP.add_column(table, name, column) -> Job

Add a column named `name` with values `column` to `table`.
The length of `column` must match the number of rows in `table`.

See also [`table_hcat`](@ref), [`add_var_column`](@ref), [`add_obs_column`](@ref).
"""
add_column(table, name, column) = Impl.add_column_job(table, name, column)

"""
    SCP.table_hcat(a, tables...) -> Job

Horizontally concatenate tables. All tables must have the same number of rows and
matching row order.

See also [`table_leftjoin`](@ref), [`add_column`](@ref).
"""
table_hcat(a, args...) = Impl.table_hcat_job(a, args...)

"""
    SCP.table_leftjoin(a, b) -> Job

Left-join table `b` onto table `a` by their ID columns.

See also [`table_hcat`](@ref), [`annotate_var`](@ref), [`annotate_obs`](@ref).
"""
table_leftjoin(a, b) = Impl.table_leftjoin_job(a, b)

"""
    SCP.transform_annotation(f, table; kwargs...) -> Job

Apply function `f` element-wise to the value column of `table`, returning a new table
with transformed values. The table must have exactly two columns (ID and value).
Use `new_name` to rename the value column.

(TODO: Example.)
"""
transform_annotation(f, table; kwargs...) = Impl.transform_annotation_job(f, table; kwargs...)
