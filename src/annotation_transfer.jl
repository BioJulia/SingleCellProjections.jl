"""
    SCP.transfer_annotation(base, new, covariate; k, kwargs...) -> Job

Transfer cell annotations from `base` to `new` using kNN-based label transfer. The
`covariate` specifies which annotation column to transfer. `k` is the number of nearest
neighbors used for voting.

Returns a table with the transferred labels and confidence scores.

(TODO: Add example - maybe I need to construct one? It should be about celltype transfer.)
"""
transfer_annotation(base, new, covariate; k, kwargs...) =
	Impl.transfer_annotation_job(base, new, covariate; k, kwargs...)
