abstract type Action end
struct Eval <: Action end
struct Projection <: Action
	replacements::Vector{Pair} # Make concrete somehow? Or just use Vector{Any} to avoid copying the vector at construction?
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



function try_replace_spec_single(spec::Spec, ::Any, k::Spec, v)
	if spec.ro == k.ro
		if v isa Spec
			return Spec(v.ro, spec.op) # Keep the op
		else
			return v # Replaced by a value, the op doesn't apply anymore
		end
	end

	return nothing
end

function try_replace_spec(spec::Spec, f::F, args...) where F
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
	Spec(p_spec.ro, spec.op) # Keep the op
end
function do_replacement(replacements, x)
	for (k,v) in replacements
		k == x && return v # replaced
	end
	return x # not replaced
end


function (proj::Projection)(x)
	# unsafe_unmanage is OK since we are only reading from proj_args and proj_kwargs
	x = ReproducibleJobs.unsafe_unmanage(x)
	ReproducibleJobs.copy_nested(y->do_replacement(proj.replacements, y), x)
end




function project(onto, p::Projectable{F}, args...) where F
	replaced = try_replace_spec(onto, onto.f, args...)
	replaced !== nothing && return replaced

	# Not found in replacements, perform projection
	res = p.f(Projection(collect(args)), onto.args...; onto.kwargs...)
	if res isa Spec
		return Spec(res.ro, onto.op) # Keep the op
	else
		return res
	end
end


project(onto, args...; kwargs...) = project(onto, onto.f, args...; kwargs...)

create_project_spec(onto, args...; kwargs...) =
	create_spec(Preprocess(project), onto, args...; kwargs...)

function Jobs.project(onto, args...; kwargs...)
	Job(create_project_spec(onto, args...; kwargs...))
end


