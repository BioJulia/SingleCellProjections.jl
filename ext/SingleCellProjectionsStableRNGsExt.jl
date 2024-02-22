module SingleCellProjectionsStableRNGsExt

using SingleCellProjections
isdefined(Base, :get_extension) ? (using StableRNGs) : (using ..StableRNGs)

SingleCellProjections.seed2rng(seed) = StableRNG(seed)

end
