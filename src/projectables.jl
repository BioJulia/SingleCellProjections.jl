abstract type Action end
struct Eval <: Action end
struct Projection <: Action
	replacements::Vector{Pair} # Make concrete somehow? Or just use Vector{Any} to avoid copying the vector at construction? Or make it as a view into the `project` node args to avoid copying altogether?
end





# is_projectable_job(x) = x isa SpecRef && x.f isa Projectable



(::Eval)(x) = x



function (p::Projectable{F})(args...; kwargs...) where F
	p.f(Eval(), args...; kwargs...)
end


# Testing ProjectOnto
function (p::ProjectOnto{F})(replacements, args...; kwargs...) where F
	project_onto_impl(p.f, replacements, args...; kwargs...)
end




function try_replace_spec_single(spec::SpecRef, ::Any, k::SpecRef, v)
	if ReproducibleJobs.get_sr(spec) === ReproducibleJobs.get_sr(k) # Because of deduplication we can use ===
		if v isa SpecRef
			return ReproducibleJobs.apply_op(k.op, v) # Transfer the op
		else
			return v # Replaced by a value, the op doesn't apply anymore
		end
	end

	return nothing
end

function try_replace_job(spec::SpecRef, f::F, args...) where F
	# @info "try_replace_job"
	# @show f
	# f isa ProjectOnto && @show f

	for (k,v) in args
		if k isa SpecRef
			r = try_replace_spec_single(spec, f, k, v)
			r !== nothing && return r
		end
	end

	return nothing
end




function do_replacement(replacements, spec::SpecRef)
	p_job = create_project_job(spec, replacements...)
	ReproducibleJobs.apply_op(spec.op, p_job) # Keep the op
end
function do_replacement(replacements, x)
	for (k,v) in replacements
		isequal(k, x) && return v # replaced
	end
	return nothing # not replaced
end


function (proj::Projection)(x)
	ReproducibleJobs.map_specs(y->do_replacement(proj.replacements, y), x)
end




# function project_impl(f::F, onto, args...) where F
# 	replaced = try_replace_job(onto, f, args...)
# 	replaced !== nothing && return replaced

# 	# Not found in replacements, apply the `Projection` action recursively
# 	p = Projection(collect(args))
# 	proj_args = p(onto.ro.value.args)
# 	proj_kwargs = p(onto.ro.value.kwargs)
# 	create_job(f, proj_args...; proj_kwargs...)
# end

# Testing with ProjectOnto
function project_impl(f::F, onto, args...) where F
	replaced = try_replace_job(onto, f, args...)
	replaced !== nothing && return replaced

	# Not found in replacements, setup as `ProjectOnto`
	replacements = (args...,)
	create_job(ProjectOnto(f), replacements, onto.args...; onto.kwargs...)
end

function project_onto_impl(f::F, replacements, args...; kwargs...) where F
	# apply the `Projection` action recursively, and keep the outer function as is
	p = Projection(collect(replacements))
	proj_args = p(args)
	proj_kwargs = p(kwargs)
	create_job(f, proj_args...; proj_kwargs...)
end



# Testing with ProjectOnto
function project_impl(p::Projectable{F}, onto, args...) where F
	replaced = try_replace_job(onto, p, args...)
	replaced !== nothing && return replaced

	# Not found in replacements, setup as `ProjectOnto`
	replacements = (args...,)
	create_job(ProjectOnto(p), replacements, onto.args...; onto.kwargs...)
end

function project_onto_impl(p::Projectable{F}, replacements, args...; kwargs...) where F
	# Perform projection
	p.f(Projection(collect(replacements)), args...; kwargs...)
	# Do we still need to keep the op? Probably not. It is handled elsewhere.
end



# Wrapper that dispatches based on the type of `onto.f`
project(::Preprocessing, onto::SpecRef, args...; kwargs...) = project_impl(onto.f, onto, args...; kwargs...)

function project(::Preprocessing, onto, args...; kwargs...)
	# Due to forwarding, `onto` might be a value.
	# TODO: Is there any reasonable use case where we would want to replace the value? I don't think so, it would have been replaced earlier.
	onto # Keep the value as is
end

# create_project_job(onto, args...; kwargs...) =
# 	create_job(Preprocess(project), onto, args...; kwargs...)

function create_project_job(onto::SpecRef, args...; kwargs...)
	# Transfer the op from `onto` to `project`
	onto2 = SpecRef(ReproducibleJobs.get_sr(onto)) # remove op
	spec = create_job(Preprocess(project), onto2, args...; kwargs...)
	ReproducibleJobs.apply_op(onto.op, spec) # apply op again
end

"""
    Jobs.project(onto, old => new, ...; kwargs...) -> Job

Projects a dataset onto another, while replacing old=>new. Multiple replacement pairs can be specified.
(TODO: Describe projection properly.)

# Examples

Given a force layout Job `fl`, we here project `proj_raw_counts` onto that force layout, by replacing `raw_counts` with `proj_raw_counts`.
```julia
julia> Jobs.project(fl, raw_counts=>proj_raw_counts)
```
"""
function Jobs.project(onto, args...; kwargs...)
	create_project_job(onto, args...; kwargs...)
end
