using BayesianDataFusion
using Base.Test

X = rand(1000, 50)
y = rand(50, 3)
lambda = 0.75

b2  = solve_full(X'*X, y, lambda)
be2 = (X'*X + eye(size(X,2)) * lambda) \ y

@test_approx_eq b2 be2

######## test general AtA_mul_B! #######
A = rand(500, 20)
x = rand(20)
y = ones(20)

AtA_mul_B!(y, A, x, 0.5)
ye = (A'*A + eye(20)*0.5)*x
@test_approx_eq y ye
