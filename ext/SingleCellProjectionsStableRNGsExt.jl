module SingleCellProjectionsStableRNGsExt

using SingleCellProjections
using .SingleCellProjections.SingleCellProjectionsCore
isdefined(Base, :get_extension) ? (using StableRNGs) : (using ..StableRNGs)

SingleCellProjectionsCore.seed2rng(seed) = StableRNG(seed)

end
