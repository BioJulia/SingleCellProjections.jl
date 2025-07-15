# Why is this needed? Probably because of SingletonType.
StableHashTraits.transformer(::Type{Projectable{F}}) where F =
	StableHashTraits.Transformer(x->x.f) # NB: pick_fields(:f) doesn't work.

ReproducibleJobs.is_preprocessing(::Projectable) = true
Base.show(io::IO, p::Projectable{F}) where F = print(io, p.f)


abstract type Action end
struct Eval <: Action end
struct Projection <: Action
	replacements::Dict{Any,Any}
end





is_projectable_spec(::Any) = false
function is_projectable_spec(sa::SpecArgs)
	f = sa.f
	f isa Projectable && return true
	# TODO: Are there more cases that should return true?
	return false
end
is_projectable_spec(spec::Spec) = is_projectable_spec(spec.ro.value)






(::Eval)(x) = x


function do_replacement(replacements, x)
	if x isa Spec
		y = get(replacements, x.ro, nothing)
		if y !== nothing
			if y isa ReproducibleJobs.ReadOnly{ReproducibleJobs.SpecArgs}
				return Spec(y, x.op) # Keep the op
			else
				return y
			end
		end
		return _setup_projection(replacements, x) # Not replaced, we need to project recursively.
	end

	y = get(replacements, x, nothing)
	y !== nothing && return y

	if x isa DataFrame
		error("No replacement provided for DataFrame with columns $(names(x)).")
	else
		return x
	end
end


function (proj::Projection)(x)
	# unsafe_unmanage is OK since we are only reading from proj_args and proj_kwargs
	x = ReproducibleJobs.unsafe_unmanage(x)
	ReproducibleJobs.copy_nested(y->do_replacement(proj.replacements, y), x)
end


function (p::Projectable{F})(args...; kwargs...) where F
	p.f(Eval(), args...; kwargs...)
end

# This might be removed since p.f can just be called directly (when refactoring is complete)
function setup_projection(replacements, p::Projectable{F}, spec::Spec) where F
	res = p.f(Projection(replacements), spec.args...; spec.kwargs...)

	if res isa Spec
		return Spec(res.ro, spec.op) # Keep the op
	else
		return res
	end
end





# TODO: Rename this function?
function _setup_projection(replacements, spec::Spec)
	p = spec.f::Union{<:Projectable,<:DataMatrixFunc}
	setup_projection(replacements, p, spec)
end


unmanage_key(p::Pair) = ReproducibleJobs.unsafe_unmanage(p[1]) => p[2]

function _split_datamatrix_replacements(replacements::Vector{<:Pair})
	out = Pair[] # to complicated to predict type

	for (k,v) in replacements
		k_is_dm = is_datamatrix_spec(k)
		v_is_dm = is_datamatrix_spec(v)
		@assert k_is_dm == v_is_dm "Both old and new must be DataMatrices if one of them is."

		if k_is_dm
			push!(out, get_matrix_spec(k).ro=>get_matrix_spec(v).ro)
			push!(out, get_var_spec(k).ro=>get_var_spec(v).ro)
			push!(out, get_obs_spec(k).ro=>get_obs_spec(v).ro)
		else
			k2 = k isa Spec ? k.ro : k
			v2 = v isa Spec ? v.ro : v
			push!(out, k2=>v2)
		end
	end
	out
end

function project(onto, args...)
	replacement_pairs = [unmanage_key(a) for a in args]
	replacement_pairs = _split_datamatrix_replacements(replacement_pairs)
	replacements = Dict(replacement_pairs)

	_setup_projection(replacements, onto)
end
ReproducibleJobs.is_preprocessing(::typeof(project)) = true

function create_project_spec(onto, args...; kwargs...)
	onto = forwarded(is_projectable_or_datamatrix_spec, onto)
	create_spec(project, onto, args...; kwargs...)
end

function Jobs.project(onto, args...; kwargs...)
	Job(create_project_spec(onto, args...; kwargs...))
end


