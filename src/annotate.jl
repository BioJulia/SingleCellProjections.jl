# TODO: Find a better name?
function annot_leftjoin_impl(annot::DataFrame, df)
	id_col = only(names(annot,1))
	df_id_col = only(names(df,1))
	@assert id_col == df_id_col "Annotation IDs didn't match, got \"$df_id_col\", but expected \"$id_col\"."
	annot = copy(annot; copycols=false)
	leftjoin!(annot, df; on=id_col)
end
annot_leftjoin(action::Action, args...) =
	create_spec(annot_leftjoin_impl, action(args)...; __use_cache=false, __version=v"0.1.0")

create_annot_leftjoin_spec(annot, df) = create_spec(Projectable(annot_leftjoin), annot, df)



annotate(::Mat, data; kwargs...) = get_matrix_spec(data)
function annotate(f::Union{Var,Obs}, data; kwargs...)
	s = get_spec(f, data)
	df = get(kwargs, f isa Var ? :var : :obs, nothing)
	df === nothing && return s
	return create_annot_leftjoin_spec(s, df)
end

# These should perhaps have a parameter saying how projections should be handled.
# Because modifications to base var we probably also want in proj var.
# Or maybe this will be solved differently, some projection step might choose to replace proj var with base var, and then it's not a problem.
Jobs.annotate_obs(data, df; kwargs...) =
	Job(create_spec(DataMatrixFunc(annotate), data; kwargs..., obs=df))
Jobs.annotate_var(data, df; kwargs...) =
	Job(create_spec(DataMatrixFunc(annotate), data; kwargs..., var=df))
Jobs.annotate(data, var_df, obs_df; kwargs...) =
	Job(create_spec(DataMatrixFunc(annotate), data; kwargs..., var=var_df, obs=obs_df))




var_counts_fraction_impl(action::Action, counts, sub_ind, tot_ind) =
	create_spec(SCPCore.counts_fraction, action(counts), action(sub_ind), action(tot_ind); dims=1, __use_cache=true, __version=v"0.1.0")
create_var_counts_fraction_impl_spec(counts, sub_ind, tot_ind) = create_spec(Projectable(var_counts_fraction_impl), counts, sub_ind, tot_ind)

var_counts_fraction(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
var_counts_fraction(::Var, counts, args...; kwargs...) = get_var_spec(counts)
function var_counts_fraction(::Obs, counts, sub_filter, tot_filter, col; project_ids)
	sub_ind = prefetch(_filter_ind(Var(), counts; fvar=sub_filter, project_var_ids=project_ids))
	tot_ind = prefetch(_filter_ind(Var(), counts; fvar=tot_filter, project_var_ids=project_ids))

	values_spec = create_var_counts_fraction_impl_spec(get_matrix_spec(counts), sub_ind, tot_ind)
	create_add_column_spec(get_obs_spec(counts), col, values_spec)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_fraction(counts, sub_filter, tot_filter, col; project_ids=:intersect)
	Job(create_spec(DataMatrixFunc(var_counts_fraction), counts, sub_filter, tot_filter, col; project_ids))
end


var_counts_sum_impl(action::Action, f, counts, ind) =
	create_spec(SCPCore.counts_sum, f, action(counts), action(ind); dims=1, __use_cache=true, __version=v"0.1.0")
create_var_counts_sum_impl_spec(f, counts, ind) = create_spec(Projectable(var_counts_sum_impl), f, counts, ind)

var_counts_sum(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
var_counts_sum(::Var, counts, args...; kwargs...) = get_var_spec(counts)
function var_counts_sum(::Obs, counts, filter, col; project_ids, f=identity)
	ind = prefetch(_filter_ind(Var(), counts; fvar=filter, project_var_ids=project_ids))
	values_spec = create_var_counts_sum_impl_spec(f, get_matrix_spec(counts), ind)
	create_add_column_spec(get_obs_spec(counts), col, values_spec)
end

# TODO: project_ids should it be :yes or :intersect by default???
function Jobs.var_counts_sum(f, counts, filter, col; project_ids=:intersect)
	Job(create_spec(DataMatrixFunc(var_counts_sum), counts, filter, col; f, project_ids))
end
Jobs.var_counts_sum(counts, filter, col; kwargs...) = Jobs.var_counts_sum(identity, counts, filter, col; kwargs...)





# TODO: Can we get better code reuse with the `var` functions above
obs_counts_fraction_impl(action::Action, counts, sub_ind, tot_ind) =
	create_spec(SCPCore.counts_fraction, action(counts), action(sub_ind), action(tot_ind); dims=2, __use_cache=true, __version=v"0.1.0")
create_obs_counts_fraction_impl_spec(counts, sub_ind, tot_ind) = create_spec(Projectable(obs_counts_fraction_impl), counts, sub_ind, tot_ind)

obs_counts_fraction(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
function obs_counts_fraction(::Var, counts, sub_filter, tot_filter, col; project_ids)
	sub_ind = prefetch(_filter_ind(Obs(), counts; fobs=sub_filter, project_obs_ids=project_ids))
	tot_ind = prefetch(_filter_ind(Obs(), counts; fobs=tot_filter, project_obs_ids=project_ids))
	values_spec = create_obs_counts_fraction_impl_spec(get_matrix_spec(counts), sub_ind, tot_ind)
	create_add_column_spec(get_var_spec(counts), col, values_spec)
end
obs_counts_fraction(::Obs, counts, args...; kwargs...) = get_obs_spec(counts)

function Jobs.obs_counts_fraction(counts, sub_filter, tot_filter, col; project_ids=:no)
	Job(create_spec(DataMatrixFunc(obs_counts_fraction), counts, sub_filter, tot_filter, col; project_ids))
end


obs_counts_sum_impl(action::Action, f, counts, ind) =
	create_spec(SCPCore.counts_sum, f, action(counts), action(ind); dims=2, __use_cache=true, __version=v"0.1.0")
create_obs_counts_sum_impl_spec(f, counts, ind) = create_spec(Projectable(obs_counts_sum_impl), f, counts, ind)

obs_counts_sum(::Mat, counts, args...; kwargs...) = get_matrix_spec(counts)
function obs_counts_sum(::Var, counts, filter, col; project_ids, f=identity)
	ind = prefetch(_filter_ind(Obs(), counts; fobs=filter, project_obs_ids=project_ids))
	values_spec = create_obs_counts_sum_impl_spec(f, get_matrix_spec(counts), ind)
	create_add_column_spec(get_var_spec(counts), col, values_spec)
end
obs_counts_sum(::Obs, counts, args...; kwargs...) = get_obs_spec(counts)

function Jobs.obs_counts_sum(f, counts, filter, col; project_ids=:no)
	Job(create_spec(DataMatrixFunc(obs_counts_sum), counts, filter, col; f, project_ids))
end
Jobs.obs_counts_sum(counts, filter, col; kwargs...) = Jobs.obs_counts_sum(identity, counts, filter, col; kwargs...)
