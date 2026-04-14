colsum!(f, dest::AbstractVector, A::AbstractMatrix) = sum!(f, reshape(dest,1,:), A; init=false)
colsum!(dest, A) = colsum(identity, dest, A)
colsum(A) = colsum(identity, A)

rowsum!(f, dest::AbstractVector, A::AbstractMatrix) = sum!(f, dest, A; init=false)
rowsum!(dest, A) = rowsum(identity, dest, A)
rowsum(A) = rowsum(identity, A)
