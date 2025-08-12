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
function is_datamatrix_spec(sa::SpecArgs)
	f = sa.f
	f isa DataMatrixFunc && return true
	f == SCPCore.DataMatrix && return true
	if f == project
		onto = sa.args[1]
		return is_datamatrix_spec(onto)
	end
	# TODO: Are there more cases that should return true?
	return false
end
is_datamatrix_spec(spec::Spec) = is_datamatrix_spec(spec.ro.value)


is_projectable_or_datamatrix_spec(x) = is_projectable_spec(x) || is_datamatrix_spec(x)



create_datamatrix_spec(matrix, var, obs) = create_spec(SCPCore.DataMatrix, matrix, var, obs; __use_cache=false, __version=v"0.1.0")



# These are needed by get_matrix etc.
setup_datamatrix(::Mat, ::typeof(SCPCore.DataMatrix), spec) = spec.args[1]
setup_datamatrix(::Var, ::typeof(SCPCore.DataMatrix), spec) = spec.args[2]
setup_datamatrix(::Obs, ::typeof(SCPCore.DataMatrix), spec) = spec.args[3]

# This changes get_field(project()) to project(get_field())
function setup_datamatrix(f::DataMatrixField, ::typeof(project), spec)
	onto = get_spec(f, spec.args[1])
	create_project_spec(onto, spec.args[2:end]...)
end



# WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
setup_datamatrix(f::DataMatrixField, d::DataMatrixFunc{F}, spec) where F = d.f(f, spec.args...; spec.kwargs...)



function get_matrix(action::Action, dm_spec)
	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_matrix
	action(setup_datamatrix(Mat(), dm_spec))
end
get_matrix_spec(x) = create_spec(Projectable(get_matrix), forwarded(is_datamatrix_spec, x))
Jobs.get_matrix(x) = Job(get_matrix_spec(x))

function get_var(action::Action, dm_spec)
	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_var
	action(setup_datamatrix(Var(), dm_spec))
end
get_var_spec(x) = create_spec(Projectable(get_var), forwarded(is_datamatrix_spec, x))
Jobs.get_var(x) = Job(get_var_spec(x))

function get_obs(action::Action, dm_spec)
	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_obs
	action(setup_datamatrix(Obs(), dm_spec))
end
get_obs_spec(x) = create_spec(Projectable(get_obs), forwarded(is_datamatrix_spec, x))
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
	create_datamatrix_spec(matrix, var, obs)
end




# TODO: Merge these 3 functions
function try_replace_spec_single(spec::Spec, p::Projectable{typeof(get_matrix)}, k::Spec, v)
	if is_datamatrix_spec(k)
		# Replace the inner spec
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_matrix_spec(res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end

function try_replace_spec_single(spec::Spec, p::Projectable{typeof(get_var)}, k::Spec, v)
	if is_datamatrix_spec(k)
		# Replace the inner spec
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_var_spec(res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end

function try_replace_spec_single(spec::Spec, p::Projectable{typeof(get_obs)}, k::Spec, v)
	if is_datamatrix_spec(k)
		# Replace the inner spec
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_obs_spec(res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end


function project(onto, d::DataMatrixFunc, args...)
	matrix = create_project_spec(get_matrix_spec(onto), args...)
	var = create_project_spec(get_var_spec(onto), args...)
	obs = create_project_spec(get_obs_spec(onto), args...)
	create_datamatrix_spec(matrix, var, obs)
end
