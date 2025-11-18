@testset "Tables" begin
	n = 10

	col_args = ("id"=>string.("id_",1:n),
	            "x"=>(1:n).^2,
	            "y"=>string.('A':'A'+n-1))

	basic_table = Jobs.create_table(col_args...)
	df = DataFrame(col_args...)

	csv_filename = tempname(; suffix=".csv")
	CSV.write(csv_filename, df)
	csv_table = Jobs.load_csv(csv_filename)



	col_args_right = ("id"=>string.("id_",n:-3:1),
	                  "z"=>n .- (n:-3:1),
	                  "z2"=> (n:-3:1) .+ 0.5)
	basic_table_right = Jobs.create_table(col_args_right...)
	df_right = DataFrame(col_args_right...)

	csv_filename_right = tempname(; suffix=".csv")
	CSV.write(csv_filename_right, df_right)
	csv_table_right = Jobs.load_csv(csv_filename_right)

	

	@testset "$name" for (name,table) in (("DataFrame",df), ("create_table",basic_table), ("CSV",csv_table))
		if table !== df
			@test isequal(fetch!(table), df)
		end

		@test fetch!(Jobs.get_colnames(table)) == ["id", "x", "y"]
		@test fetch!(Jobs.get_id_colname(table)) == "id"
		@test fetch!(Jobs.id_column(table)) == select(df, "id"; copycols=false)
		@test fetch!(Jobs.table_nrow(table)) == n

		let cols = ["y", "id"]
			@test isequal(fetch!(Jobs.get_columns(table, cols...)), select(df, cols))
		end
		let cols = ["y", "not_a_column_name", "id"]
			@test_throws ReproducibleJobs.ProcessingException fetch!(Jobs.get_columns(table, cols...))
		end

		@test isequal(fetch!(Jobs.column_data(table, "id")), df.id)
		@test isequal(fetch!(Jobs.column_data(table, "x")), df.x)
		@test isequal(fetch!(Jobs.column_data(table, "y")), df.y)
		@test_throws ReproducibleJobs.ProcessingException fetch!(Jobs.column_data(table, "not_a_column_name"))


		annot = Jobs.annotation(table, "y")
		
		@test fetch!(Jobs.get_id_colname(annot)) == "id"
		@test fetch!(Jobs.get_value_colname(annot)) == "y"
		@test_throws ReproducibleJobs.ProcessingException fetch!(Jobs.get_value_colname(table))

		@test fetch!(Jobs.id_column(annot)) == select(df, "id")
		@test fetch!(Jobs.value_column(annot)) == select(df, "y")
		@test_throws ReproducibleJobs.ProcessingException fetch!(Jobs.value_column(table))

		@test fetch!(Jobs.id_column_data(annot)) == df[!,"id"]
		@test fetch!(Jobs.value_column_data(annot)) == df[!,"y"]
		@test_throws ReproducibleJobs.ProcessingException fetch!(Jobs.value_column_data(table))


		let u_data = sqrt.(1:n)
			df2 = insertcols(df, "u"=>u_data)
			table2 = Jobs.add_column(table, "u", u_data)
			@test isequal(fetch!(table2), df2)
		end

		let ind = collect(n:-2:1)
			table2 = ReproducibleJobs.Job(SingleCellProjections.table_getindex_spec(table, ind))
			@test isequal(fetch!(table2), df[ind,:])
		end

		@testset "Leftjoin with $name" for (name,table_right) in (("DataFrame",df_right), ("create_table",basic_table_right), ("CSV",csv_table_right))
			table2 = Jobs.table_leftjoin(table, table_right)
			@test isequal(fetch!(table2), leftjoin(df, df_right; on=:id, order=:left))
		end
	end


	# TODO: This is slightly inconvenient, since we need to replace filenames at the TimestampedFilePath level. Can we get around that somehow?
	@testset "ReplaceFilename" begin
		csv_table_p = Jobs.project(csv_table, TimestampedFilePath(csv_filename)=>TimestampedFilePath(csv_filename_right))
		@test isequal(forward(csv_table_p).spec, forward(csv_table_right).spec)
	end

	# TODO: Projections
	# Important! Use specs that resolve at different levels
	# * create_table spec with Projectables as column data - i.e. projection is done at the column level
	# * create_table spec after projcetion                 - i.e. projection is done before the column level
	# * DataFrame                                          - i.e. we hit the fallback

end
