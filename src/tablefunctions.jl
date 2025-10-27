abstract type TableField end # Find a better name?
struct ColNames <: TableField end
struct Col <: TableField
	name::String
end



create_table_impl(args::Pair...) = DataFrame(args...)
create_table_impl_spec(args::Pair...) = create_spec(create_table_impl, args...; __version=v"0.0.1")

create_table_pr(action::Action, args::Pair...) = create_table_impl_spec(action(args)...)
create_table_pr_spec(args::Pair...) = create_spec(Projectable(create_table_pr), args...)




# These are needed by get_colnames/get_col
setup_table(::ColNames, ::typeof(DataFrame), spec) =
	String[k for (k,v) in spec.args]
function setup_table(c::Col, ::typeof(DataFrame), spec)
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
setup_table(f::TableField, t::TableFunction{F}, spec) where F = t.f(f, spec.args...; spec.kwargs...) # Should this be ColNames only?
setup_table(col::Col, t::ColNamesTableFunction{F}, spec) where F = t.f(col, spec.args...; spec.kwargs...)


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
	f isa ColNamesTableFunction && return true
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
	cols = (name=>f(Col(name), args...; kwargs...) for name in colnames)
	create_table_impl_spec(cols...)
end



# for dispatch
setup_table(f::TableField, spec::Spec) = setup_table(f, spec.f, spec)



# This evaluates the TableFunction
function (d::TableFunction{F})(args...; kwargs...) where F
	colnames = d.f(ColNames(), args...; kwargs...) # can be a spec or just a list of names

	# We need to preprocess once to fetch the colnames.
	# This also means that colnames can be Projected.
	create_spec(ColNamesTableFunction(d.f), fetched(colnames), args...; kwargs...)

	# TODO: Should we do this? It's just a small shortcut for when colnames are not a Spec. But it should work fine with projections too.
	# if colnames isa Union{Spec,Job}
	# 	# We need to preprocess once to fetch the colnames
	# 	create_spec(ColNamesTableFunction(d.f), fetched(colnames), args...; kwargs...)
	# else
	# 	# colnames are not a spec, just setup directly
	# 	_table_from_colnames(d.f, colnames, args...; kwargs...)
	# end
end


function (d::ColNamesTableFunction{F})(colnames, args...; kwargs...) where F
	_table_from_colnames(d.f, colnames, args...; kwargs...)
end





function try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_colnames)}, k::Spec, v)
	if is_table_spec(k)
		# Replace the inner spec
		res = try_replace_spec_single(spec.args[1], nothing, k, v)
		return res === nothing ? res : get_spec(ColNames(), res)
	else
		# Fallback to standard replace
		return try_replace_spec_single(spec, nothing, k, v)
	end
end
function try_replace_spec_single(spec::Spec, ::Projectable{typeof(get_col)}, k::Spec, v)
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
	onto2 = create_spec(ColNamesTableFunction(t.f), onto.args...; onto.kwargs...)
	create_project_spec(onto2, args...; colnames)
end

function project(onto, s::ColNamesTableFunction, args...; colnames)
	# Given the column names, project the columns
	cols = (name=>create_project_spec(get_col(onto, name), args...) for name in colnames)
	create_table_impl_spec(cols...)
end
