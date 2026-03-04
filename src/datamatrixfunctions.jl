is_datamatrix_spec(::Any) = false
function is_datamatrix_spec(sa::SpecArgs)
	f = sa.f
	f isa DataMatrixFunction && return true
	f == SCPCore.DataMatrix && return true
	f isa ProjectOnto{<:DataMatrixFunction} && return true # testing ProjectOnto
	if f == project # TODO: Remove?
		onto = sa.args[1]
		return is_datamatrix_spec(onto)
	end
	# TODO: Are there more cases that should return true?
	return false
end
is_datamatrix_spec(spec::Spec) = is_datamatrix_spec(spec.sa)




create_datamatrix_spec(matrix, var, obs) = create_spec(SCPCore.DataMatrix, matrix, var, obs; __version=v"0.1.0")



# These are needed by get_matrix etc.
setup_datamatrix(::Mat, ::typeof(SCPCore.DataMatrix), spec) = spec.args[1]
setup_datamatrix(::Var, ::typeof(SCPCore.DataMatrix), spec) = spec.args[2]
setup_datamatrix(::Obs, ::typeof(SCPCore.DataMatrix), spec) = spec.args[3]

# This changes get_field(project()) to project(get_field())
function setup_datamatrix(f::DataMatrixField, ::typeof(project), spec)
	onto = get_spec(f, spec.args[1])
	create_project_spec(onto, spec.args[2:end]...)
end

# TODO: Testing with ProjectOnto
# This changes get_field(ProjectOnto()) to project(get_field())
function setup_datamatrix(f::DataMatrixField, p::ProjectOnto{<:DataMatrixFunction}, spec)
	onto_dm = create_spec(p.f, spec.args[2:end]...; spec.kwargs...) # recreate original spec...
	onto = get_spec(f, onto_dm)
	create_project_spec(onto, spec.args[1]...)
end



# WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
setup_datamatrix(f::DataMatrixField, d::DataMatrixFunction{F}, spec) where F = d.f(f, spec.args...; spec.kwargs...)


# # Hopefully deprecated
# function get_matrix_pr(action::Action, dm_spec)
# 	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_matrix
# 	action(setup_datamatrix(Mat(), dm_spec))
# end
# function get_var_pr(action::Action, dm_spec)
# 	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_var
# 	action(setup_datamatrix(Var(), dm_spec))
# end
# function get_obs_pr(action::Action, dm_spec)
# 	@assert is_datamatrix_spec(dm_spec) # TODO: We might want to relax this later, and instead call SCPCore.get_obs
# 	action(setup_datamatrix(Obs(), dm_spec))
# end


# get_matrix(::Preprocessing, dm_spec) = create_spec(Projectable(get_matrix_pr), dm_spec)
# get_var(::Preprocessing, dm_spec) = create_spec(Projectable(get_var_pr), dm_spec)
# get_obs(::Preprocessing, dm_spec) = create_spec(Projectable(get_obs_pr), dm_spec)

# New test of improved preprocessing
# get_matrix(::Preprocessing, dm_spec) = create_spec(MatFunction(dm_spec.f.f), dm_spec.args...; dm_spec.kwargs...)
# get_var(::Preprocessing, dm_spec) = create_spec(VarFunction(dm_spec.f.f), dm_spec.args...; dm_spec.kwargs...)
# get_obs(::Preprocessing, dm_spec) = create_spec(ObsFunction(dm_spec.f.f), dm_spec.args...; dm_spec.kwargs...)

# New test
# TODO: Improve code reuse
function get_matrix(::Preprocessing, dm)
	if dm isa DataMatrix
		return dm.matrix
	end

	f = dm.f
	if f isa ProjectOnto
		# @info "get_matrix ProjectOnto"

		# TODO: Avoid this construct
		onto = create_spec(f.f, dm.args[2:end]...; dm.kwargs...) # recreate original spec...
		replaced = try_replace_spec(onto, f.f, dm.args[1]...)
		replaced !== nothing && return replaced

		create_spec(ProjectOnto(MatFunction(f.f.f)), dm.args...; dm.kwargs...)
	elseif f isa DataMatrixFunction
		create_spec(MatFunction(f.f), dm.args...; dm.kwargs...)
	else
		error("get_matrix cannot be used on a $(typeof(f))")
	end
end
function get_var(::Preprocessing, dm)
	if dm isa DataMatrix
		return dm.var
	end

	f = dm.f
	if f isa ProjectOnto
		# @info "get_var ProjectOnto"

		# TODO: Avoid this construct
		onto = create_spec(f.f, dm.args[2:end]...; dm.kwargs...) # recreate original spec...
		replaced = try_replace_spec(onto, f.f, dm.args[1]...)
		replaced !== nothing && return replaced

		create_spec(ProjectOnto(VarFunction(f.f.f)), dm.args...; dm.kwargs...)
	elseif f isa DataMatrixFunction
		create_spec(VarFunction(f.f), dm.args...; dm.kwargs...)
	else
		error("get_var cannot be used on a $(typeof(f))")
	end
end
function get_obs(::Preprocessing, dm)
	if dm isa DataMatrix
		return dm.obs
	end

	f = dm.f
	if f isa ProjectOnto
		# @info "get_obs ProjectOnto"

		# TODO: Avoid this construct
		onto = create_spec(f.f, dm.args[2:end]...; dm.kwargs...) # recreate original spec...
		replaced = try_replace_spec(onto, f.f, dm.args[1]...)
		replaced !== nothing && return replaced

		create_spec(ProjectOnto(ObsFunction(f.f.f)), dm.args...; dm.kwargs...)
	elseif f isa DataMatrixFunction
		create_spec(ObsFunction(f.f), dm.args...; dm.kwargs...)
	else
		error("get_obs cannot be used on a $(typeof(f))")
	end
end

# # DEBUG version
# get_matrix(::Preprocessing, dm_spec) = (@show dm_spec; @info Mat(); create_spec(MatFunction(dm_spec.f.f), dm_spec.args...; dm_spec.kwargs...))
# get_var(::Preprocessing, dm_spec) = (@show dm_spec; @info Var(); create_spec(VarFunction(dm_spec.f.f), dm_spec.args...; dm_spec.kwargs...))
# get_obs(::Preprocessing, dm_spec) = (@show dm_spec; @info Obs(); create_spec(ObsFunction(dm_spec.f.f), dm_spec.args...; dm_spec.kwargs...))


# # Testing improved preprocessing
# # Doesn't really work with replacements, so we skip it for now...
# function get_matrix(::Preprocessing, dm_spec)
# 	if is_datamatrix_spec(dm_spec)
# 		setup_datamatrix(Mat(), dm_spec)
# 	else
# 		# TODO: Get rid of this when we have fixed the projection case
# 		create_spec(Projectable(get_matrix_pr), dm_spec)
# 	end
# end
# function get_var(::Preprocessing, dm_spec)
# 	if is_datamatrix_spec(dm_spec)
# 		setup_datamatrix(Var(), dm_spec)
# 	else
# 		# TODO: Get rid of this when we have fixed the projection case
# 		create_spec(Projectable(get_var_pr), dm_spec)
# 	end
# end
# function get_obs(::Preprocessing, dm_spec)
# 	if is_datamatrix_spec(dm_spec)
# 		setup_datamatrix(Obs(), dm_spec)
# 	else
# 		# TODO: Get rid of this when we have fixed the projection case
# 		create_spec(Projectable(get_obs_pr), dm_spec)
# 	end
# end


get_matrix_spec(x) = create_spec(Preprocess(get_matrix), x)
Jobs.get_matrix(x) = Job(get_matrix_spec(x))

get_var_spec(x) = create_spec(Preprocess(get_var), x)
Jobs.get_var(x) = Job(get_var_spec(x))

get_obs_spec(x) = create_spec(Preprocess(get_obs), x)
Jobs.get_obs(x) = Job(get_obs_spec(x))

get_spec(::Mat, x) = get_matrix_spec(x)
get_spec(::Var, x) = get_var_spec(x)
get_spec(::Obs, x) = get_obs_spec(x)





# for dispatch
setup_datamatrix(f::DataMatrixField, spec::Spec) = setup_datamatrix(f, spec.f, spec)




# This evaluates the DataMatrixFunction by wrapping the subspecs in a DataMatrix
function (d::DataMatrixFunction{F})(args...; kwargs...) where F
	matrix = d.f(Mat(), args...; kwargs...)
	var = d.f(Var(), args...; kwargs...)
	obs = d.f(Obs(), args...; kwargs...)
	create_datamatrix_spec(matrix, var, obs)
end

# This evaluates a single field of a DataMatrixFunction
function (d::DataMatrixFieldFunction{T,F})(args...; kwargs...) where {T,F}
	# @show d
	d.f(T(), args...; kwargs...)
end




function _try_replace_get_spec_single(f::DataMatrixField, spec::Spec, k::Spec, v)
	# @info "_try_replace_get_spec_single"
	if is_datamatrix_spec(k)
		# TODO: Improve this code, avoid recreating the original spec
		onto = create_spec(DataMatrixFunction(spec.f.f), spec.args...; spec.kwargs...) # recreate original spec...
		res = try_replace_spec_single(onto, nothing, k, v)
		return res === nothing ? res : get_spec(f, res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end

# try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_matrix_pr)}, k::Spec, v) =
# 	_try_replace_get_spec_single(Mat(), spec, k, v)
# try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_var_pr)}, k::Spec, v) =
# 	_try_replace_get_spec_single(Var(), spec, k, v)
# try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_obs_pr)}, k::Spec, v) =
# 	_try_replace_get_spec_single(Obs(), spec, k, v)


# # Testing ProjectOnto
# try_replace_spec_single(spec::Spec, ::ProjectOnto{<:MatFunction}, k::Spec, v) =
# 	_try_replace_get_spec_single(Mat(), spec, k, v)
# try_replace_spec_single(spec::Spec, ::ProjectOnto{<:VarFunction}, k::Spec, v) =
# 	_try_replace_get_spec_single(Var(), spec, k, v)
# try_replace_spec_single(spec::Spec, ::ProjectOnto{<:ObsFunction}, k::Spec, v) =
# 	_try_replace_get_spec_single(Obs(), spec, k, v)

# Testing ProjectOnto
try_replace_spec_single(spec::Spec, ::MatFunction, k::Spec, v) =
	_try_replace_get_spec_single(Mat(), spec, k, v)
try_replace_spec_single(spec::Spec, ::VarFunction, k::Spec, v) =
	_try_replace_get_spec_single(Var(), spec, k, v)
try_replace_spec_single(spec::Spec, ::ObsFunction, k::Spec, v) =
	_try_replace_get_spec_single(Obs(), spec, k, v)


# function project_impl(::DataMatrixFunction, onto, args...)
# 	matrix = create_project_spec(get_matrix_spec(onto), args...)
# 	var = create_project_spec(get_var_spec(onto), args...)
# 	obs = create_project_spec(get_obs_spec(onto), args...)
# 	create_datamatrix_spec(matrix, var, obs)
# end

# Testing with ProjectOnto
function project_impl(d::DataMatrixFunction, onto, args...)
	# @info "project_impl(d::DataMatrixFunction)"
	replaced = try_replace_spec(onto, d, args...)
	replaced !== nothing && return replaced

	# Setup as `ProjectOnto`
	replacements = (args...,)
	create_spec(ProjectOnto(d), replacements, onto.args...; onto.kwargs...)
end

function project_onto_impl(d::DataMatrixFunction{F}, replacements, args...; kwargs...) where F
	onto = create_spec(d, args...; kwargs...) # recreate original spec...
	matrix = create_project_spec(get_matrix_spec(onto), replacements...)
	var = create_project_spec(get_var_spec(onto), replacements...)
	obs = create_project_spec(get_obs_spec(onto), replacements...)
	create_datamatrix_spec(matrix, var, obs)
end


# Testing with ProjectOnto
function project_impl(d::DataMatrixFieldFunction, onto, args...)
	# @info "project_impl(d::DataMatrixFieldFunction)"
	replaced = try_replace_spec(onto, d, args...)
	replaced !== nothing && return replaced

	# Setup as `ProjectOnto`
	replacements = (args...,)
	create_spec(ProjectOnto(d), replacements, onto.args...; onto.kwargs...)
end

function project_onto_impl(d::DataMatrixFieldFunction{T,F}, replacements, args...; kwargs...) where {T,F}
	# @info "project_onto_impl(d::DataMatrixFieldFunction)"
	p = Projection(collect(replacements))
	# Setup data matrix field and then perform projection

	# @show F
	# @show replacements
	# @show args
	# @show kwargs

	p(d.f(T(), args...; kwargs...))
end
