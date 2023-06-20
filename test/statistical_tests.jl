@testset "Statistical Tests" begin
	P,N = (50,587)

	c = copy(counts)
	c.obs.group2 = replace(c.obs.group, "C"=>missing)

	l = logtransform(c)
	tf = tf_idf_transform(c)

	@testset "A vs B" begin
		test_ind = 47
		# ground truth
		x = convert(Vector, l.matrix.matrix[test_ind,:])
		mw = ApproximateMannWhitneyUTest(x[isequal.("A",l.obs.group2)], x[isequal.("B",l.obs.group2)])

		@testset "$f" for f in (mannwhitney_table, mannwhitney, mannwhitney!)
			data = f==mannwhitney! ? copy(l) : l

			for (args,prefix) in ((("group","A","B"), "group_A_vs_B_"),
			                      (("group2",),       "group2_"),
			                      (("group2","A"),    "group2_A_"))
				result = f(data, args...)

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
			end
		end
	end

	@testset "B vs Other" begin
		test_ind = 23
		# ground truth
		x = convert(Vector, tf.matrix.matrix[test_ind,:])
		mw = ApproximateMannWhitneyUTest(x[isequal.("B",tf.obs.group)], x[.!isequal.("B",tf.obs.group)])

		@testset "$f" for f in (mannwhitney_table, mannwhitney, mannwhitney!)
			data = f==mannwhitney! ? copy(tf) : tf

			result = f(data, "group", "B")

			u_col = "U"
			p_col = "pValue"
			if f==mannwhitney_table
				df = result
			else
				df = result.var
				u_col = string("group_B_",u_col)
				p_col = string("group_B_",p_col)

				# columns should only be added to source if using mannwhitney!
				@test hasproperty(data.var, u_col) == (f==mannwhitney!)
				@test hasproperty(data.var, p_col) == (f==mannwhitney!)
			end

			@test df[test_ind,u_col] == mw.U
			@test df[test_ind,p_col] ≈ pvalue(mw)
		end
	end
	
	@testset "column names" begin
		test_ind = 19
		# ground truth
		x = convert(Vector, c.matrix[test_ind,:])
		mw = ApproximateMannWhitneyUTest(x[isequal.("C",c.obs.group)], x[.!isequal.("C",c.obs.group)])

		df = mannwhitney_table(c, "group", "C"; statistic_col="my_u", pvalue_col="my_p")
		@test df[test_ind,"my_u"] == mw.U
		@test df[test_ind,"my_p"] ≈ pvalue(mw)

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