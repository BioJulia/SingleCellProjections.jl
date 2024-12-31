module SingleCellProjectionsStableHashTraitsExt

using SingleCellProjections
isdefined(Base, :get_extension) ? (using StableHashTraits) : (using ..StableHashTraits)

# This is needed since StatelessModel is treated as StructTypes.SingletonType() when F<:Function
StableHashTraits.transformer(::Type{<:StatelessModel{F}}) where F =
	StableHashTraits.Transformer(x::StatelessModel{F}->x.f) # NB: pick_fields(:f) doesn't work

end
