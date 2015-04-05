using BayesianDataFusion
using Base.Test

X = rand(1000, 50)
y = rand(50, 3)
lambda = 0.75

b1 = solve_cg(X, y, lambda)
b2 = solve_full(X'*X, y, lambda)

@test_approx_eq b1 b2
