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
	f == create_table_impl && return true
	if f == project
		onto = sa.args[1]
		return is_table_spec(onto)
	end
	# TODO: Are there more cases that should return true?
	return false
end
is_table_spec(spec::Spec) = is_table_spec(spec.ro.value)




# for dispatch
setup_table(f::TableField, spec::Spec) = setup_table(f, spec.f, spec)

# This evaluates the TableFunction - Step 1, figure out colnames
function (t::TableFunction{F})(args...; kwargs...) where F
	colnames = wrap_colnames_spec(t.f(ColNames(), args...; kwargs...))
	create_spec(t, fetched(colnames), args...; kwargs...)
end
# This evaluates the TableFunction - Step 2, create the table
function (t::TableFunction{F})(colnames::ColNameVector, args...; kwargs...) where F
	colnames = unwrap_colnames(colnames)
	cols = (name=>t.f(Col(name), args...; kwargs...) for name in colnames)
	create_table_impl_spec(cols...)
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
