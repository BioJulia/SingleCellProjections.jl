col_sum_squared(X) = compute(DiagGram(matrixexpression(X)))
row_sum_squared(X) = compute(DiagGram(matrixexpression(X)'))

sum_squared_to_var(x, n) = max.(0, x) ./ (n-1)
