abstract type TableField end # Find a better name?
struct ColNames <: TableField end
struct Col <: TableField
	name::String
end


# For dispatch reasons, we need this to be its own type and not just a Vector.
# Users should never have to interact with it.
struct ColNameVector{T} # T should be an AbstractVector or a ReadOnly
	v::T
end


function wrap_colnames(colnames)
	ColNameVector(colnames)
end
wrap_colnames_spec(colnames) = create_spec(wrap_colnames, colnames; __version=v"0.0.1")

# _unwrap_colnames(c::ColNameVector{ReadOnly{T}}) where T = c.v.value
# _unwrap_colnames(c::ColNameVector{T}) where T = c.v
# ReproducibleJobs.unmanage_rec(c::ColNameVector{<:ReadOnly{<:Vector}}) = ReadOnlyVector(c.v.value)

unwrap_colnames(c::ColNameVector{ReadOnly{T}}) where T<:Vector = ReadOnlyVector(c.v.value)
ReproducibleJobs.unmanage_rec(c::ColNameVector) = unwrap_colnames(c)


# Is something like this needed?
# ReproducibleJobs.copy_arg(x::ColNameVector) = ColNameVector(copy_arg(x.v))


ReproducibleJobs.copy_nested(f, c::ColNameVector) = f(ColNameVector(ReproducibleJobs.copy_nested(f, c.v)))
function ReproducibleJobs.visit_nested(f, pred, c::ColNameVector)
	pred(c.v) && visit_nested(f, pred, c.v)
end



create_table_impl(args::Pair...) = DataFrame(args...)
create_table_impl_spec(args::Pair...) = create_spec(create_table_impl, args...; __version=v"0.0.1")

create_table_pr(action::Action, args::Pair...) = create_table_impl_spec(action(args)...)
create_table_pr_spec(args::Pair...) = create_spec(Projectable(create_table_pr), args...)




# These are needed by get_colnames/get_col
setup_table(::ColNames, ::typeof(create_table_impl), spec) =
	String[k for (k,v) in spec.args]
function setup_table(c::Col, ::typeof(create_table_impl), spec)
	for (k,v) in spec.args
		k == c.name && return v
	end
	throw(KeyError(c.name))
end


# This changes get_field(project()) to project(get_field())
function setup_table(f::TableField, ::typeof(project), spec)
	onto = get_spec(f, spec.args[1])
	create_project_spec(onto, spec.args[2:end]...)
end



# # WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
# setup_table(f::TableField, t::TableFunction{F}, spec) where F = t.f(f, spec.args...; spec.kwargs...) # Should this be ColNames only?
# setup_table(col::Col, t::ColNamesTableFunction{F}, spec) where F = t.f(col, spec.args...; spec.kwargs...)






# # WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
# function setup_table(::ColNames, t::TableFunction{F}, spec) where F
# 	@info "setup_table(::ColNames)"
# 	s = t.f(ColNames(), spec.args...; spec.kwargs...)
# 	wrap_colnames_spec(s)
# end
# setup_table(col::Col, t::TableFunction{F}, spec) where F = t.f(col, spec.args...; spec.kwargs...) # Should this be removed?
# setup_table(col::Col, t::ColNamesTableFunction{F}, spec) where F = t.f(col, spec.args...; spec.kwargs...)




# # WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
# function setup_table(::ColNames, t::TableFunction{F}, spec) where F
# 	@info "setup_table(::ColNames)"

# 	# sanity checks - we do not expected to come here if we have already computed the ColNameVector
# 	if length(spec.args)>1
# 		a = first(spec.args)
# 		@assert !(a isa ColNameVector)
# 		@assert !(a isa Spec) || a.f != wrap_colnames
# 	end


# 	s = t.f(ColNames(), spec.args...; spec.kwargs...)
# 	wrap_colnames_spec(s)
# end
# function setup_table(col::Col, t::TableFunction{F}, spec) where F
# 	@info "hej"
# 	@show spec.args

# 	# # sanity checks - we only expect to come here after we have computed the ColNameVector
# 	# @assert length(spec.args)>=1
# 	# a = first(spec.args)
# 	# @show typeof(a)
# 	# @assert a isa ColNameVector || (a isa Spec && a.f == wrap_colnames)
# 	# Should we @assert that col.name is in the ColNameVector?


# 	# sanity checks - we do not expected to come here if we have already computed the ColNameVector
# 	if length(spec.args)>1
# 		a = first(spec.args)
# 		@assert !(a isa ColNameVector)
# 		@assert !(a isa Spec) || a.f != wrap_colnames
# 	end


# 	t.f(col, spec.args...; spec.kwargs...)
# 	# t.f(col, spec.args[2:end]...; spec.kwargs...)
# end



# WIP - perhaps spec should be unwrapped at an earlier point - perhaps in ReproducibleJobs?
function setup_table(f::TableField, t::TableFunction{F}, spec) where F
	# sanity checks - we do not expected to come here if we have already computed the ColNameVector
	if length(spec.args)>1
		a = first(spec.args)
		@assert !(a isa ColNameVector)
		@assert !(a isa Spec) || a.f != wrap_colnames
	end
	s = t.f(f, spec.args...; spec.kwargs...)
	if f isa ColNames
		wrap_colnames_spec(s)
	else
		s
	end
end



function get_colnames_pr(action::Action, table_spec)
	@assert is_table_spec(table_spec) # TODO: We might want to relax this later
	action(setup_table(ColNames(), table_spec))
end
function get_col_pr(action::Action, table_spec, name)
	@assert is_table_spec(table_spec) # TODO: We might want to relax this later
	action(setup_table(Col(name), table_spec))
end


get_colnames(table_spec) = create_spec(Projectable(get_colnames_pr), table_spec)
get_col(table_spec, name) = create_spec(Projectable(get_col_pr), table_spec, name)

get_colnames_spec(x) = create_spec(Preprocess(get_colnames), x)
Jobs.get_colnames(x) = Job(get_colnames_spec(x))

get_col_spec(x, name) = create_spec(Preprocess(get_col), x, name)
Jobs.get_col(x, name) = Job(get_col_spec(x, name))

get_spec(::ColNames, x) = get_colnames_spec(x)
get_spec(c::Col, x) = get_col_spec(x, c.name)




is_table_spec(::Any) = false
function is_table_spec(sa::SpecArgs)
	f = sa.f
	f isa TableFunction && return true
	# f isa ColNamesTableFunction && return true
	f == create_table_impl && return true
	if f == project
		onto = sa.args[1]
		return is_table_spec(onto)
	end
	# TODO: Are there more cases that should return true?
	return false
end
is_table_spec(spec::Spec) = is_table_spec(spec.ro.value)



function _table_from_colnames(f::F, colnames, args...; kwargs...) where F
	# @show typeof(colnames)
	# if colnames isa ReadOnly
	# 	colnames = colnames.value
	# end
	# if colnames isa ColNameVector{<:ReadOnly} # make this nicer?
	# 	colnames = colnames.v.value
	# end
	colnames = unwrap_colnames(colnames)

	cols = (name=>f(Col(name), args...; kwargs...) for name in colnames)
	create_table_impl_spec(cols...)
end



# for dispatch
setup_table(f::TableField, spec::Spec) = setup_table(f, spec.f, spec)



# # This evaluates the TableFunction
# function (d::TableFunction{F})(args...; kwargs...) where F
# 	# colnames = d.f(ColNames(), args...; kwargs...) # can be a spec or just a list of names
# 	colnames = wrap_colnames_spec(d.f(ColNames(), args...; kwargs...))

# 	# We need to preprocess once to fetch the colnames.
# 	# This also means that colnames can be Projected.
# 	create_spec(ColNamesTableFunction(d.f), fetched(colnames), args...; kwargs...)

# 	# TODO: Should we do this? It's just a small shortcut for when colnames are not a Spec. But it should work fine with projections too.
# 	# if colnames isa Union{Spec,Job}
# 	# 	# We need to preprocess once to fetch the colnames
# 	# 	create_spec(ColNamesTableFunction(d.f), fetched(colnames), args...; kwargs...)
# 	# else
# 	# 	# colnames are not a spec, just setup directly
# 	# 	_table_from_colnames(d.f, colnames, args...; kwargs...)
# 	# end
# end

# function (d::ColNamesTableFunction{F})(colnames, args...; kwargs...) where F
# 	_table_from_colnames(d.f, colnames, args...; kwargs...)
# end


# This evaluates the TableFunction
function (d::TableFunction{F})(args...; kwargs...) where F
	colnames = wrap_colnames_spec(d.f(ColNames(), args...; kwargs...))
	# We need to preprocess once to fetch the colnames.
	# This also means that colnames can be Projected.
	create_spec(d, fetched(colnames), args...; kwargs...)
end
function (d::TableFunction{F})(colnames::ColNameVector, args...; kwargs...) where F
	_table_from_colnames(d.f, colnames, args...; kwargs...)
end





function try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_colnames_pr)}, k::Spec, v)
	if is_table_spec(k)
		# Replace the inner spec
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_spec(ColNames(), res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end
function try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_col_pr)}, k::Spec, v)
	if is_table_spec(k)
		# Replace the inner spec
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_spec(Col(spec.args[2]), res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end




# function project(onto, t::TableFunction, args...; colnames=nothing)
# 	if colnames === nothing
# 		# Project the column names
# 		colnames = fetched(create_project_spec(get_colnames(onto), args...))
# 		create_project_spec(onto, args...; colnames)
# 	else
# 		# if colnames isa ReadOnly
# 		# 	colnames = colnames.value
# 		# end
# 		# if colnames isa ColNameVector{<:ReadOnly} # make this nicer?
# 		# 	colnames = colnames.v.value
# 		# end
# 		colnames = unwrap_colnames(colnames)

# 		# Given the column names, project the columns
# 		cols = (name=>create_project_spec(get_col(onto, name), args...) for name in colnames)
# 		create_table_impl_spec(cols...)
# 	end
# end



function project(onto, t::TableFunction, args...)
	# Project the column names
	colnames = fetched(create_project_spec(get_colnames(onto), args...))
	create_project_spec(onto, colnames, args...)
end
function project(onto, t::TableFunction, colnames::ColNameVector, args...)
	colnames = unwrap_colnames(colnames)

	# Given the column names, project the columns
	cols = (name=>create_project_spec(get_col(onto, name), args...) for name in colnames)
	create_table_impl_spec(cols...)
end




# function project(onto, t::TableFunction, args...)
# 	# Project the column names
# 	colnames = fetched(create_project_spec(get_colnames(onto), args...))
# 	onto2 = create_spec(ColNamesTableFunction(t.f), onto.args...; onto.kwargs...)
# 	create_project_spec(onto2, args...; colnames)
# end

# function project(onto, s::ColNamesTableFunction, args...; colnames)
# 	# Given the column names, project the columns
# 	cols = (name=>create_project_spec(get_col(onto, name), args...) for name in colnames)
# 	create_table_impl_spec(cols...)
# end
