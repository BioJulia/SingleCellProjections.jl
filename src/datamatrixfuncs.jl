# Why is this needed? Probably because of SingletonType.
StableHashTraits.transformer(::Type{DataMatrixFunc{F}}) where F =
	StableHashTraits.Transformer(x->x.f) # NB: pick_fields(:f) doesn't work.

ReproducibleJobs.is_preprocessing(::DataMatrixFunc) = true
Base.show(io::IO, d::DataMatrixFunc{F}) where F = print(io, d.f)

abstract type DataMatrixField end
struct Mat <: DataMatrixField end
struct Var <: DataMatrixField end
struct Obs <: DataMatrixField end



is_datamatrix_spec(::Any) = false
function is_datamatrix_spec(spec::Spec)
	f = spec.f
	f isa DataMatrixFunc && return true
	f == SCPCore.DataMatrix && return true
	if f == project
		onto = spec.args[1]
		return is_datamatrix_spec(onto)
	end
	# TODO: Are there more cases that should return true?
	return false
end


# These are needed by get_matrix etc.
setup_datamatrix(::Mat, ::typeof(SCPCore.DataMatrix), spec) = spec.args[1]
setup_datamatrix(::Var, ::typeof(SCPCore.DataMatrix), spec) = spec.args[2]
setup_datamatrix(::Obs, ::typeof(SCPCore.DataMatrix), spec) = spec.args[3]

# This is needed when replacing with something that itself is a projection
setup_datamatrix(f::DataMatrixField, ::typeof(project), spec) = setup_datamatrix(f, project(spec)) # is this right?


# WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
setup_datamatrix(f::DataMatrixField, d::DataMatrixFunc{F}, spec) where F = d.f(f, spec.args...; spec.kwargs...)



function get_matrix(action::Action, dm_spec)
	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_matrix
	action(setup_datamatrix(Mat(), dm_spec))
end
get_matrix_spec(x) = create_spec(Projectable(get_matrix), x)
Jobs.get_matrix(x) = Job(get_matrix_spec(x))

function get_var(action::Action, dm_spec)
	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_var
	action(setup_datamatrix(Var(), dm_spec))
end
get_var_spec(x) = create_spec(Projectable(get_var), x)
Jobs.get_var(x) = Job(get_var_spec(x))

function get_obs(action::Action, dm_spec)
	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_obs
	action(setup_datamatrix(Obs(), dm_spec))
end
get_obs_spec(x) = create_spec(Projectable(get_obs), x)
Jobs.get_obs(x) = Job(get_obs_spec(x))


get_spec(::Mat, x) = get_matrix_spec(x)
get_spec(::Var, x) = get_var_spec(x)
get_spec(::Obs, x) = get_obs_spec(x)





# for dispatch
setup_datamatrix(f::DataMatrixField, spec::Spec) = setup_datamatrix(f, spec.f, spec)




# This evaluates the DataMatrixFunc by wrapping the subspecs in a DataMatrix
function (d::DataMatrixFunc{F})(args...; kwargs...) where F
	matrix = d.f(Mat(), args...; kwargs...)
	var = d.f(Var(), args...; kwargs...)
	obs = d.f(Obs(), args...; kwargs...)
	create_spec(SCPCore.DataMatrix, matrix, var, obs; __use_cache=false, __version=v"0.1.0")
end

# General projection handling
function setup_projection(replacements, ::DataMatrixFunc, spec::Spec)
	# Replacements work at the get_xyz level
	matrix_spec = get_matrix_spec(spec)
	var_spec = get_var_spec(spec)
	obs_spec = get_obs_spec(spec)

	# TODO: Should these be here? (Definitely needed right now.)
	mr = get(replacements, matrix_spec, nothing)
	matrix_spec = @something mr _setup_projection(replacements, matrix_spec)
	vr = get(replacements, var_spec, nothing)
	var_spec = @something vr _setup_projection(replacements, var_spec)
	or = get(replacements, obs_spec, nothing)
	obs_spec = @something or _setup_projection(replacements, obs_spec)

	create_spec(SCPCore.DataMatrix, matrix_spec, var_spec, obs_spec; __use_cache=false, __version=v"0.1.0")
end
