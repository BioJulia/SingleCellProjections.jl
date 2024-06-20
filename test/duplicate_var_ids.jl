@testset "Duplicate var IDs" begin
	@testset "$f" for f in (load10x, load_counts)
		@test_throws ErrorException f(rna_adt_h5_path; duplicate_var=:error)
		c = f(rna_adt_h5_path; duplicate_var=:warn)
		c2 = f(rna_adt_h5_path; duplicate_var=:ignore)
		@test c==c2

		id_ans = ["NEXN", "CD2", "CD48", "TOR1AIP2", "CDC73", "CD55", "AL391597.1", "CD46", "CD34", "TMEM63A", "MERTK", "CD28", "MREG", "CD47", "CD86", "SLC49A3", "AC093871.1", "MIR3945HG", "LINC02220", "TSLP", "CD14", "CD109", "LINC01611", "CD24", "LINC02587", "CD36", "GAGE12B", "AC064807.1", "AC027031.2", "AC090159.1", "CD59", "CD9", "CD69", "ASCL1", "TRAJ3", "U91319.1", "IGHV3OR16-13", "CCR10", "CRHR1", "CLTC", "ZNF236", "AL109954.2", "CD40", "APC2", "CD209", "LINC00662", "CD22", "ZNF234", "SLC6A16", "KIR3DX1", "CD34", "CD36", "CD22", "CD24", "CD209", "CD284", "CD66a/c/e", "CD14", "TCR-1", "CD140b", "CD305", "CD2", "CD9", "CD193", "CD86", "CCR10", "CD275-2", "CD69", "CD102", "CD109", "TCR-V-24-J-18", "CD55", "CD59", "CD243", "MERTK", "CD48", "CD47", "CD46", "CD40", "CD28"]
		ft_ans = vcat(fill("Gene Expression",50), fill("Antibody Capture",30))
		var_id_ans = string.(id_ans, '_', ft_ans)

		mat_sum_dim1_ans = [583 646 563 700 602 654 658 551 888 1104 512 900 687 393 704 910 563 446 405 683 775 994 767 466 758 387 798 634 666 2295 549 746 1465 877 2069 418 804 1007 1842 1036 1585 541 2086 729 590 847 1012 718 2599 884 720 545 1063 1765 859 590 706 766 839 1453 492 1333 735 606 914 4004 503 550 762 5349 566 1235 705 686 871 1127 621 1088 1039 1542 1601 841 1082 451 804 298 597 671 386 1075 327 617 353 384 785 411 141 480 327 428 613 1119 479 749 293 723 565 855 474 295 611 1030 397 686 478 700 589 568 641 635 440 686 484 407 672 723 544 455 963 428 372 536 895 486 914 439 672 1519 391 939 1010 679 430 1254 944 676 432 903 907 888 409 834 627 476 516 1373 735 268 170 935 601 472 1056 428 503 444 1287 571 574 503 532 559 779 540 677 688 526 516 503 1133 423 664 420 289 562 924 1240 400 856 429 770 492 539 721 495 1407 862 164 751 576]
		mat_sum_dim2_ans = [10; 292; 835; 31; 58; 139; 0; 135; 6; 35; 5; 32; 6; 171; 80; 5; 0; 0; 0; 1; 393; 4; 0; 1; 0; 149; 0; 7; 2; 0; 27; 25; 156; 0; 0; 0; 0; 7; 0; 82; 20; 0; 11; 0; 1; 15; 14; 7; 1; 1; 2733; 1210; 3611; 1031; 1174; 287; 586; 4364; 1204; 484; 6288; 10019; 7911; 913; 3514; 2905; 1875; 3512; 16307; 6474; 280; 6872; 1319; 1755; 3136; 40000; 13134; 3359; 289; 6686;;]

		@test c.var.id == id_ans
		@test c.var.feature_type == ft_ans
		@test sum(c.matrix; dims=1) == mat_sum_dim1_ans
		@test sum(c.matrix; dims=2) == mat_sum_dim2_ans

		c = f(rna_adt_h5_path; var_id="var_id"=>["id","feature_type"], duplicate_var=:error)
		@test c.var.id == id_ans
		@test c.var.feature_type == ft_ans
		@test c.var.var_id == var_id_ans
		@test sum(c.matrix; dims=1) == mat_sum_dim1_ans
		@test sum(c.matrix; dims=2) == mat_sum_dim2_ans
	end


	c = load10x(rna_adt_h5_path; duplicate_var=:ignore)
	cRNA = load10x(rna_h5_path; duplicate_var=:error)

	@testset "logtransform" begin
		l = logtransform(c)
		l2 = logtransform(cRNA)
		p = project(c,l2)
		p2 = project(cRNA,l)

		@testset "$name" for (name,a,b) in (("l",l,l2),("proj1",p,l2),("proj2",l,p2))
			@test a.matrix == b.matrix
			@test a.obs == b.obs
			@test sort(names(a.var)) == sort(names(b.var)) # var should be equal
			@test a.var == select(b.var, names(a.var))     # (up to order of columns)
		end
	end

	@testset "tf_idf_transform" begin
		tf = tf_idf_transform(c)
		tf2 = tf_idf_transform(cRNA)

		p = project(c,tf2)
		p2 = project(cRNA,tf)

		@testset "$name" for (name,a,b) in (("tf",tf,tf2),("proj1",p,tf2),("proj2",tf,p2))
			@test a.matrix == b.matrix
			@test a.obs == b.obs
			@test sort(names(a.var)) == sort(names(b.var)) # var should be equal
			@test a.var == select(b.var, names(a.var))     # (up to order of columns)
		end
	end

	@testset "SCTransform" begin
		s = sctransform(c; use_cache=false)
		s2 = sctransform(cRNA; use_cache=false)
		p = project(c,s2)
		p2 = project(cRNA,s)

		@testset "$name" for (name,a,b) in (("s",s,s2),("proj1",p,s2),("proj2",s,p2))
			@test a.matrix == b.matrix
			@test a.obs == b.obs
			@test sort(names(a.var)) == sort(names(b.var)) # var should be equal
			@test a.var == select(b.var, names(a.var))     # (up to order of columns)
		end
	end
end
