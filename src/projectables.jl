"""
    SCP.project(onto, old => new, ...; kwargs...) -> Job

Projects a dataset onto another, while replacing old=>new. Multiple replacement pairs can be specified.
(TODO: Describe projection properly.)

# Examples

Given a force layout Job `fl`, we here project `proj_raw_counts` onto that force layout, by replacing `raw_counts` with `proj_raw_counts`.
```julia
julia> SCP.project(fl, raw_counts=>proj_raw_counts)
```
"""
function project(onto, args...; kwargs...)
	Impl.create_project_job(onto, args...; kwargs...)
end
