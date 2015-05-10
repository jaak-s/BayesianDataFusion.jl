using BayesianDataFusion
using Base.Test

A = rand(500, 20)
x = rand(20)
y = ones(20)

######## test general AtA_mul_B! #######
AtA_mul_B!(y, A, x, 0.5)
ye = (A'*A + eye(20)*0.5)*x
@test_approx_eq y ye
