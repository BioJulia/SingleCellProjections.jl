abstract type Action end
struct Eval <: Action end
struct Projection <: Action
	replacements::Vector{Pair} # Make concrete somehow? Or just use Vector{Any} to avoid copying the vector at construction? Or make it as a view into the `project` node args to avoid copying altogether?
end





is_projectable_spec(x) = x isa Spec && x.f isa Projectable

# is_projectable_spec(::Any) = false
# function is_projectable_spec(sa::SpecArgs)
# 	f = sa.f
# 	f isa Projectable && return true
# 	# TODO: Are there more cases that should return true?
# 	return false
# end
# is_projectable_spec(spec::Spec) = is_projectable_spec(spec.ro.value)





(::Eval)(x) = x



function (p::Projectable{F})(args...; kwargs...) where F
	p.f(Eval(), args...; kwargs...)
end


# Testing ProjectOnto
function (p::ProjectOnto{F})(replacements, args...; kwargs...) where F
	project_onto_impl(p.f, replacements, args...; kwargs...)
end



function try_replace_spec_single(spec::Spec, ::Any, k::Spec, v)
	if spec.sa === k.sa # Because of deduplication we can use ===
		if v isa Spec
			return Spec(v.sa, spec.op) # Keep the op
		else
			return v # Replaced by a value, the op doesn't apply anymore
		end
	end

	return nothing
end

function try_replace_spec(spec::Spec, f::F, args...) where F
	# @info "try_replace_spec"
	# @show f
	# f isa ProjectOnto && @show f

	for (k,v) in args
		if k isa Spec
			r = try_replace_spec_single(spec, f, k, v)
			r !== nothing && return r
		end
	end

	return nothing
end




function do_replacement(replacements, spec::Spec)
	p_spec = create_project_spec(spec, replacements...)
	Spec(p_spec.sa, spec.op) # Keep the op
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
# 	replaced = try_replace_spec(onto, f, args...)
# 	replaced !== nothing && return replaced

# 	# Not found in replacements, apply the `Projection` action recursively
# 	p = Projection(collect(args))
# 	proj_args = p(onto.ro.value.args)
# 	proj_kwargs = p(onto.ro.value.kwargs)
# 	create_spec(f, proj_args...; proj_kwargs...)
# end

# Testing with ProjectOnto
function project_impl(f::F, onto, args...) where F
	replaced = try_replace_spec(onto, f, args...)
	replaced !== nothing && return replaced

	# Not found in replacements, setup as `ProjectOnto`
	replacements = (args...,)
	create_spec(ProjectOnto(f), replacements, onto.args...; onto.kwargs...)
end

function project_onto_impl(f::F, replacements, args...; kwargs...) where F
	# apply the `Projection` action recursively, and keep the outer function as is
	p = Projection(collect(replacements))
	proj_args = p(args)
	proj_kwargs = p(kwargs)
	create_spec(f, proj_args...; proj_kwargs...)
end



# function project_impl(p::Projectable{F}, onto, args...) where F
# 	replaced = try_replace_spec(onto, p, args...)
# 	replaced !== nothing && return replaced

# 	# Not found in replacements, perform projection
# 	res = p.f(Projection(collect(args)), onto.args...; onto.kwargs...)
# 	if res isa Spec
# 		return Spec(res.ro, onto.op) # Keep the op
# 	else
# 		return res
# 	end
# end

# Testing with ProjectOnto
function project_impl(p::Projectable{F}, onto, args...) where F
	replaced = try_replace_spec(onto, p, args...)
	replaced !== nothing && return replaced

	# Not found in replacements, setup as `ProjectOnto`
	replacements = (args...,)
	create_spec(ProjectOnto(p), replacements, onto.args...; onto.kwargs...)
end

function project_onto_impl(p::Projectable{F}, replacements, args...; kwargs...) where F
	# Perform projection
	p.f(Projection(collect(replacements)), args...; kwargs...)

	# Do we still need to keep the op? Probably not. It is handled elsewhere.
	# if res isa Spec
	# 	return Spec(res.ro, onto.op) # Keep the op
	# else
	# 	return res
	# end
end



# Wrapper that dispatches based on the type of `onto.f`
project(::Preprocessing, onto::Spec, args...; kwargs...) = project_impl(onto.f, onto, args...; kwargs...)

function project(::Preprocessing, onto, args...; kwargs...)
	# Due to forwarding, `onto` might be a value.
	# TODO: Is there any reasonable use case where we would want to replace the value? I don't think so, it would have been replaced earlier.
	onto # Keep the value as is
end

# create_project_spec(onto, args...; kwargs...) =
# 	create_spec(Preprocess(project), onto, args...; kwargs...)

function create_project_spec(onto, args...; kwargs...)
	# Transfer the op from `onto` to `project`
	# TODO: make code cleaner

	onto isa Job && (onto = onto.spec)
	onto::Spec

	op = onto.op
	onto = Spec(onto.sa)
	spec = create_spec(Preprocess(project), onto, args...; kwargs...)
	Spec(spec.sa, op)
end

function Jobs.project(onto, args...; kwargs...)
	Job(create_project_spec(onto, args...; kwargs...))
end
