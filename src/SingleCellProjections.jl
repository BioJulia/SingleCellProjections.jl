module SingleCellProjections

export
	SingleCellProjectionsCore, # TODO: make public instead
	Jobs # TODO: remove probably


# This symbol is only defined on Julia versions that support extensions
isdefined(Base, :get_extension) || using Requires


include("SingleCellProjectionsCore/SingleCellProjectionsCore.jl")
using .SingleCellProjectionsCore




include("reproducible.jl")


# include("precompile.jl")



@static if !isdefined(Base, :get_extension)
	function __init__()
		@require UMAP="c4f8c510-2410-5be4-91d7-4fbaeb39457e" include("../ext/SingleCellProjectionsUMAPExt.jl")
		@require TSne="24678dba-d5e9-5843-a4c6-250288b04835" include("../ext/SingleCellProjectionsTSneExt.jl")
		@require PrincipalMomentAnalysis="6a3ba550-3b7f-11e9-2734-d9178ad1e8db" include("../ext/SingleCellProjectionsPrincipalMomentAnalysisExt.jl")
		@require Muon="446846d7-b4ce-489d-bf74-72da18fe3629" include("../ext/SingleCellProjectionsMuonExt.jl")
	end
end



end
