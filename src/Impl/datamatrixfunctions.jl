is_datamatrix_job(::Any) = false
function is_datamatrix_job(spec::SpecRef)
	f = spec.f
	f isa DataMatrixFunction && return true
	f == SCPCore.DataMatrix && return true
	f isa ProjectOnto{<:DataMatrixFunction} && return true # testing ProjectOnto
	if f == project # TODO: Remove?
		onto = spec.args[1]
		return is_datamatrix_job(onto)
	end
	# TODO: Are there more cases that should return true?
	return false
end




create_datamatrix_job(matrix, var, obs) = create_job(SCPCore.DataMatrix, matrix, var, obs; __version=v"0.1.0")



# These are needed by get_matrix etc.
setup_datamatrix(::Mat, ::typeof(SCPCore.DataMatrix), spec) = spec.args[1]
setup_datamatrix(::Var, ::typeof(SCPCore.DataMatrix), spec) = spec.args[2]
setup_datamatrix(::Obs, ::typeof(SCPCore.DataMatrix), spec) = spec.args[3]

# This changes get_field(project()) to project(get_field())
function setup_datamatrix(f::DataMatrixField, ::typeof(project), spec)
	onto = get_job(f, spec.args[1])
	create_project_job(onto, spec.args[2:end]...)
end

# TODO: Testing with ProjectOnto
# This changes get_field(ProjectOnto()) to project(get_field())
function setup_datamatrix(f::DataMatrixField, p::ProjectOnto{<:DataMatrixFunction}, spec)
	onto_dm = create_job(p.f, spec.args[2:end]...; spec.kwargs...) # recreate original spec...
	onto = get_job(f, onto_dm)
	create_project_job(onto, spec.args[1]...)
end



# WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
setup_datamatrix(f::DataMatrixField, d::DataMatrixFunction{F}, spec) where F = d.f(f, spec.args...; spec.kwargs...)


# # Hopefully deprecated
# function get_matrix_pr(action::Action, dm_job)
# 	@assert is_datamatrix_job(dm_job) # TODO: We might want to relax this later, and instead call SCPCore.get_matrix
# 	action(setup_datamatrix(Mat(), dm_job))
# end
# function get_var_pr(action::Action, dm_job)
# 	@assert is_datamatrix_job(dm_job) # TODO: We might want to relax this later, and instead call SCPCore.get_var
# 	action(setup_datamatrix(Var(), dm_job))
# end
# function get_obs_pr(action::Action, dm_job)
# 	@assert is_datamatrix_job(dm_job) # TODO: We might want to relax this later, and instead call SCPCore.get_obs
# 	action(setup_datamatrix(Obs(), dm_job))
# end


# get_matrix(::Preprocessing, dm_job) = create_job(Projectable(get_matrix_pr), dm_job)
# get_var(::Preprocessing, dm_job) = create_job(Projectable(get_var_pr), dm_job)
# get_obs(::Preprocessing, dm_job) = create_job(Projectable(get_obs_pr), dm_job)

# New test of improved preprocessing
# get_matrix(::Preprocessing, dm_job) = create_job(MatFunction(dm_job.f.f), dm_job.args...; dm_job.kwargs...)
# get_var(::Preprocessing, dm_job) = create_job(VarFunction(dm_job.f.f), dm_job.args...; dm_job.kwargs...)
# get_obs(::Preprocessing, dm_job) = create_job(ObsFunction(dm_job.f.f), dm_job.args...; dm_job.kwargs...)

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
		onto = create_job(f.f, dm.args[2:end]...; dm.kwargs...) # recreate original spec...
		replaced = try_replace_job(onto, f.f, dm.args[1]...)
		replaced !== nothing && return replaced

		create_job(ProjectOnto(MatFunction(f.f.f)), dm.args...; dm.kwargs...)
	elseif f isa DataMatrixFunction
		create_job(MatFunction(f.f), dm.args...; dm.kwargs...)
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
		onto = create_job(f.f, dm.args[2:end]...; dm.kwargs...) # recreate original spec...
		replaced = try_replace_job(onto, f.f, dm.args[1]...)
		replaced !== nothing && return replaced

		create_job(ProjectOnto(VarFunction(f.f.f)), dm.args...; dm.kwargs...)
	elseif f isa DataMatrixFunction
		create_job(VarFunction(f.f), dm.args...; dm.kwargs...)
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
		onto = create_job(f.f, dm.args[2:end]...; dm.kwargs...) # recreate original spec...
		replaced = try_replace_job(onto, f.f, dm.args[1]...)
		replaced !== nothing && return replaced

		create_job(ProjectOnto(ObsFunction(f.f.f)), dm.args...; dm.kwargs...)
	elseif f isa DataMatrixFunction
		create_job(ObsFunction(f.f), dm.args...; dm.kwargs...)
	else
		error("get_obs cannot be used on a $(typeof(f))")
	end
end

# # DEBUG version
# get_matrix(::Preprocessing, dm_job) = (@show dm_job; @info Mat(); create_job(MatFunction(dm_job.f.f), dm_job.args...; dm_job.kwargs...))
# get_var(::Preprocessing, dm_job) = (@show dm_job; @info Var(); create_job(VarFunction(dm_job.f.f), dm_job.args...; dm_job.kwargs...))
# get_obs(::Preprocessing, dm_job) = (@show dm_job; @info Obs(); create_job(ObsFunction(dm_job.f.f), dm_job.args...; dm_job.kwargs...))


# # Testing improved preprocessing
# # Doesn't really work with replacements, so we skip it for now...
# function get_matrix(::Preprocessing, dm_job)
# 	if is_datamatrix_job(dm_job)
# 		setup_datamatrix(Mat(), dm_job)
# 	else
# 		# TODO: Get rid of this when we have fixed the projection case
# 		create_job(Projectable(get_matrix_pr), dm_job)
# 	end
# end
# function get_var(::Preprocessing, dm_job)
# 	if is_datamatrix_job(dm_job)
# 		setup_datamatrix(Var(), dm_job)
# 	else
# 		# TODO: Get rid of this when we have fixed the projection case
# 		create_job(Projectable(get_var_pr), dm_job)
# 	end
# end
# function get_obs(::Preprocessing, dm_job)
# 	if is_datamatrix_job(dm_job)
# 		setup_datamatrix(Obs(), dm_job)
# 	else
# 		# TODO: Get rid of this when we have fixed the projection case
# 		create_job(Projectable(get_obs_pr), dm_job)
# 	end
# end


get_matrix_job(x) = create_job(Preprocess(get_matrix), x)

get_var_job(x) = create_job(Preprocess(get_var), x)

get_obs_job(x) = create_job(Preprocess(get_obs), x)

get_job(::Mat, x) = get_matrix_job(x)
get_job(::Var, x) = get_var_job(x)
get_job(::Obs, x) = get_obs_job(x)





# for dispatch
setup_datamatrix(f::DataMatrixField, spec::SpecRef) = setup_datamatrix(f, spec.f, spec)




# This evaluates the DataMatrixFunction by wrapping the subspecs in a DataMatrix
function (d::DataMatrixFunction{F})(args...; kwargs...) where F
	matrix = d.f(Mat(), args...; kwargs...)
	var = d.f(Var(), args...; kwargs...)
	obs = d.f(Obs(), args...; kwargs...)
	create_datamatrix_job(matrix, var, obs)
end

# This evaluates a single field of a DataMatrixFunction
function (d::DataMatrixFieldFunction{T,F})(args...; kwargs...) where {T,F}
	# @show d
	d.f(T(), args...; kwargs...)
end




function _try_replace_get_spec_single(f::DataMatrixField, spec::SpecRef, k::SpecRef, v)
	# @info "_try_replace_get_spec_single"
	if is_datamatrix_job(k)
		# TODO: Improve this code, avoid recreating the original spec
		onto = create_job(DataMatrixFunction(spec.f.f), spec.args...; spec.kwargs...) # recreate original spec...
		res = try_replace_spec_single(onto, nothing, k, v)
		return res === nothing ? res : get_job(f, res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end

# Testing ProjectOnto
try_replace_spec_single(spec::SpecRef, ::MatFunction, k::SpecRef, v) =
	_try_replace_get_spec_single(Mat(), spec, k, v)
try_replace_spec_single(spec::SpecRef, ::VarFunction, k::SpecRef, v) =
	_try_replace_get_spec_single(Var(), spec, k, v)
try_replace_spec_single(spec::SpecRef, ::ObsFunction, k::SpecRef, v) =
	_try_replace_get_spec_single(Obs(), spec, k, v)


# Testing with ProjectOnto
function project_impl(d::DataMatrixFunction, onto, args...)
	# @info "project_impl(d::DataMatrixFunction)"
	replaced = try_replace_job(onto, d, args...)
	replaced !== nothing && return replaced

	# Setup as `ProjectOnto`
	replacements = (args...,)
	create_job(ProjectOnto(d), replacements, onto.args...; onto.kwargs...)
end

function project_onto_impl(d::DataMatrixFunction{F}, replacements, args...; kwargs...) where F
	onto = create_job(d, args...; kwargs...) # recreate original spec...
	matrix = create_project_job(get_matrix_job(onto), replacements...)
	var = create_project_job(get_var_job(onto), replacements...)
	obs = create_project_job(get_obs_job(onto), replacements...)
	create_datamatrix_job(matrix, var, obs)
end


# Testing with ProjectOnto
function project_impl(d::DataMatrixFieldFunction, onto, args...)
	# @info "project_impl(d::DataMatrixFieldFunction)"
	replaced = try_replace_job(onto, d, args...)
	replaced !== nothing && return replaced

	# Setup as `ProjectOnto`
	replacements = (args...,)
	create_job(ProjectOnto(d), replacements, onto.args...; onto.kwargs...)
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
