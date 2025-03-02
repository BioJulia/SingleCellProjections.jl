@testset "Mann-Whitney U-test" begin
	P,N = (50,587)

	c = copy(counts)
	c.obs.group2 = replace(c.obs.group, "C"=>missing)

	l = logtransform(c)
	tf = tf_idf_transform(c)

	@testset "A vs B" begin
		X = convert(Matrix,l.matrix.matrix)
		mw = [ApproximateMannWhitneyUTest(X[i,isequal.("A",l.obs.group2)], X[i,isequal.("B",l.obs.group2)]) for i in 1:size(X,1)]
		mwU = getfield.(mw,:U)
		mwP = pvalue.(mw)

		@testset "$f" for f in (mannwhitney_table, mannwhitney, mannwhitney!)
			setup = ((("group","A","B"),                "group_A_vs_B_"),
			         (("group2",),                      "group2_"),
			         (("group2","A"),                   "group2_A_"),
			         ((covariate("group","A","B"),),    "group_A_vs_B_"),
			         ((covariate("group2",:twogroup),), "group2_"),
			         ((covariate("group2","A"),),       "group2_A_"),
			        )

			cov_str(c::CovariateDesc) = covariate_prefix(c,"'")
			cov_str(h1) = h1
			cov_str(h1,group_a) = "$(h1)_$group_a"
			cov_str(h1,group_a,group_b) = "$(h1)_$(group_a)_vs_$group_b"

			@testset "$(cov_str(h1...))" for (h1,prefix) in setup
				data = f==mannwhitney! ? copy(l) : l
				result = f(data, h1...)

				u_col = "U"
				p_col = "pValue"
				if f==mannwhitney_table
					df = result
				else
					df = result.var
					u_col = string(prefix,u_col)
					p_col = string(prefix,p_col)

					# columns should only be added to source if using mannwhitney!
					@test hasproperty(data.var, u_col) == (f==mannwhitney!)
					@test hasproperty(data.var, p_col) == (f==mannwhitney!)
				end

				@test df[:,u_col] == mwU
				@test df[:,p_col] ≈ mwP

				if any(ismissing, data.obs[:,covariate(h1...).src])
					@test_throws "missing values" f(data, h1...; h1_missing=:error)
				else
					data2 = f==mannwhitney! ? copy(l) : l
					result2 = f(data2, h1...; h1_missing=:error)
					df2 = f==mannwhitney_table ? result2 : result2.var

					@test isequal(df, df2)
				end
			end
		end
	end

	@testset "B vs Other" begin
		setup = (("group","B","group_B_"),
		         ("group2","B","group2_B_"),
		        )

		@testset "$column $value" for (column,value,prefix) in setup
			test_ind = 23
			# ground truth
			x = convert(Vector, tf.matrix.matrix[test_ind,:])

			mask1 = isequal.(value,tf.obs[!,column])
			mask2 = .!mask1 .& .!ismissing.(tf.obs[!,column])
			mw = ApproximateMannWhitneyUTest(x[mask1], x[mask2])

			@testset "$f" for f in (mannwhitney_table, mannwhitney, mannwhitney!)
				data = f==mannwhitney! ? copy(tf) : tf

				result = f(data, column, value)

				u_col = "U"
				p_col = "pValue"
				if f==mannwhitney_table
					df = result
				else
					df = result.var
					u_col = string(prefix,u_col)
					p_col = string(prefix,p_col)

					# columns should only be added to source if using mannwhitney!
					@test hasproperty(data.var, u_col) == (f==mannwhitney!)
					@test hasproperty(data.var, p_col) == (f==mannwhitney!)
				end

				@test df[test_ind,u_col] == mw.U
				@test df[test_ind,p_col] ≈ pvalue(mw)


				if any(ismissing, data.obs[:,column])
					@test_throws "missing values" f(data, column, value; h1_missing=:error)
				else
					data2 = f==mannwhitney! ? copy(tf) : tf
					result2 = f(data2, column, value; h1_missing=:error)
					df2 = f==mannwhitney_table ? result2 : result2.var

					@test isequal(df, df2)
				end
			end
		end
	end
	
	@testset "Column names" begin
		test_ind = 19
		# ground truth
		x = convert(Vector, c.matrix[test_ind,:])
		mw = ApproximateMannWhitneyUTest(x[isequal.("C",c.obs.group)], x[.!isequal.("C",c.obs.group)])

		df = mannwhitney_table(c, "group", "C"; statistic_col="my_u", pvalue_col="my_p")
		@test df[test_ind,"my_u"] == mw.U
		@test df[test_ind,"my_p"] ≈ pvalue(mw)

		df = mannwhitney_table(c, "group", "C"; statistic_col=nothing, pvalue_col="my_p")
		@test !hasproperty(df, "U")
		@test df[test_ind,"my_p"] ≈ pvalue(mw)

		df = mannwhitney_table(c, "group", "C"; statistic_col="my_u", pvalue_col=nothing)
		@test df[test_ind,"my_u"] == mw.U
		@test !hasproperty(df, "my_p")

		data = mannwhitney(c, "group", "C"; statistic_col="my_u", pvalue_col="my_p")
		@test data.var[test_ind,"my_u"] == mw.U
		@test data.var[test_ind,"my_p"] ≈ pvalue(mw)
		@test !hasproperty(c.var, "my_u")
		@test !hasproperty(c.var, "my_p")

		data = copy(c)
		mannwhitney!(data, "group", "C"; statistic_col="my_u", pvalue_col="my_p")
		@test data.var[test_ind,"my_u"] == mw.U
		@test data.var[test_ind,"my_p"] ≈ pvalue(mw)


		data = mannwhitney(c, "group", "C"; prefix="another_")
		@test data.var[test_ind,"another_U"] == mw.U
		@test data.var[test_ind,"another_pValue"] ≈ pvalue(mw)
		@test !hasproperty(c.var, "another_U")
		@test !hasproperty(c.var, "another_pValue")

		data = copy(c)
		mannwhitney!(data, "group", "C"; prefix="another_")
		@test data.var[test_ind,"another_U"] == mw.U
		@test data.var[test_ind,"another_pValue"] ≈ pvalue(mw)
	end
end
