"""
    SCP.local_outlier_factor(data, full; k=10, col="LOF") -> Job

Compute the [Local Outlier Factor](https://en.wikipedia.org/wiki/Local_outlier_factor)
for each observation in `data` relative to the full dataset `full`, using `k` nearest
neighbors. Returns a table with IDs and LOF scores in a column named `col`.

When projecting, only neighbors in the base dataset are considered.
"""
function local_outlier_factor(data, full; kwargs...)
	Impl.local_outlier_factor_job(data, full; kwargs...)
end
