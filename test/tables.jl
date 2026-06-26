using Test
using SingleCellProjections
using ReproducibleJobs: ReproducibleJobs, TimestampedFilePath, prefetched, fetch!, forward!
using CSV
using DataFrames

function run_tables_tests()
	@testset "Tables" begin
		n = 10
		n_p = 4

		col_args = ("id"=>string.("id_",1:n),
		            "x"=>(1:n).^2,
		            "y"=>string.('A':'A'+n-1))

		col_args_p = ("id"=>string.("id_",1:n_p),
		                 "x"=>(1:n_p).^3,
		                 "y"=>string.('a':'a'+n_p-1))


		basic_table = Jobs.create_table(col_args...)
		df = DataFrame(col_args...)

		csv_filename = tempname()*".csv"
		CSV.write(csv_filename, df)
		csv_table = Jobs.load_csv(csv_filename)


		basic_table_p = Jobs.create_table(col_args_p...)
		df_p = DataFrame(col_args_p...)

		csv_filename_p = tempname()*".csv"
		CSV.write(csv_filename_p, df_p)
		csv_table_p = Jobs.load_csv(csv_filename_p)


		# For hcat
		col_args_hc = ("x2"=>(1:n).^2,
		               "y2"=>string.('A':'A'+n-1))

		basic_table_hc = Jobs.create_table(col_args_hc...)
		df_hc = DataFrame(col_args_hc...)

		csv_filename_hc = tempname()*".csv"
		CSV.write(csv_filename_hc, df_hc)
		csv_table_hc = Jobs.load_csv(csv_filename_hc)



		# For leftjoin
		col_args_right = ("id"=>string.("id_",n:-3:1),
		                  "z"=>n .- (n:-3:1),
		                  "z2"=> (n:-3:1) .+ 0.5)
		basic_table_right = Jobs.create_table(col_args_right...)
		df_right = DataFrame(col_args_right...)

		csv_filename_right = tempname()*".csv"
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
			@test fetch!(Jobs.table_ncol(table)) == 3

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

			if name != "DataFrame" # TODO: Enable this test for DataFrames to when I make it work
				let annot_fwd = forward!(annot)
					@test annot_fwd.f === SingleCellProjections.create_table
					@test first.(annot_fwd.args) == ["id", "y"]
				end
			end



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
				@test fetch!(Jobs.table_ncol(table2)) == 4
			end

			let ind = collect(n:-2:1)
				table2 = SingleCellProjections.table_getindex_job(table, ind)
				@test isequal(fetch!(table2), df[ind,:])
			end

			@testset "table_hcat with $name_hc" for (name_hc,table_hc) in (("DataFrame",df_hc), ("create_table",basic_table_hc), ("CSV",csv_table_hc))
				table2 = Jobs.table_hcat(table, table_hc)
				@test isequal(fetch!(table2), hcat(df, df_hc))
			end

			@testset "Leftjoin with $name" for (name,table_right) in (("DataFrame",df_right), ("create_table",basic_table_right), ("CSV",csv_table_right))
				table2 = Jobs.table_leftjoin(table, table_right)
				@test isequal(fetch!(table2), leftjoin(df, df_right; on=:id, order=:left))
				@test fetch!(Jobs.table_ncol(table2)) == 5
			end
		end


		# TODO: This is slightly inconvenient, since we need to replace filenames at the TimestampedFilePath level. Can we get around that somehow?
		@testset "ReplaceFilename" begin
			csv_table_p = Jobs.project(csv_table, TimestampedFilePath(csv_filename)=>TimestampedFilePath(csv_filename_right))
			@test isequal(forward!(csv_table_p), forward!(csv_table_right))
		end


		@testset "transform_annotation $name" for (name,table,table_p) in (("DataFrame",df,df_p), ("create_table",basic_table,basic_table_p), ("CSV",csv_table,csv_table_p))
			annot_x = Jobs.annotation(table, "x")  # id + x (numeric)
			x_vals = Float64.(df.x)

			@testset "simple transform" begin
				ta = Jobs.transform_annotation(sqrt, annot_x)
				@test isequal(fetch!(ta), DataFrame("id"=>df.id, "x"=>sqrt.(x_vals)))
			end

			@testset "new_name" begin
				ta = Jobs.transform_annotation(sqrt, annot_x; new_name="sqrtx")
				@test isequal(fetch!(ta), DataFrame("id"=>df.id, "sqrtx"=>sqrt.(x_vals)))
			end

			@testset "prefetched scalar inside Base.Fix2" begin
				max_job = SingleCellProjections.apply_job(maximum, Jobs.value_column_data(annot_x))
				ta = Jobs.transform_annotation(Base.Fix2(/, prefetched(max_job)), annot_x)
				expected = DataFrame("id"=>df.id, "x"=>x_vals ./ maximum(x_vals))
				@test isequal(fetch!(ta), expected)

				# Projectables without any replacements
				ta_p = Jobs.project(ta)
				@test isequal(forward!(ta), forward!(ta_p))

				v = Jobs.value_column_data(ta)
				v_p = Jobs.project(v)
				@test isequal(forward!(v), forward!(v_p))
				@test isequal(fetch!(ta_p), expected)


				# TODO: Implement replacements for Tables and enable this
				# projections
				# ta_p = Jobs.project(ta, table=>table_p) # Hmm. This doesn't work. Because `table` isn't kept as a unit during projection.
				# expected_p = DataFrame("id"=>df_p.id, "x"=>df_p.x ./ maximum(df_p.x))
				# @test isequal(fetch!(ta_p), expected_p)
			end
		end


		# TODO: Projections
		# Important! Use specs that resolve at different levels
		# * create_table spec with Projectables as column data - i.e. projection is done at the column level
		# * create_table spec after projection                 - i.e. projection is done before the column level
		# * DataFrame                                          - i.e. we hit the fallback

	end
end
