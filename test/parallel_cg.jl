using BayesianDataFusion
using Base.Test

A = rand(500, 20)
x = rand(20)
y = ones(20)

######## test general AtA_mul_B! #######
AtA_mul_B!(y, A, x, 0.5)
ye = (A'*A + eye(20)*0.5)*x
@test_approx_eq y ye

cg = BayesianDataFusion.CG(A, 0.5, workers())
A_mul_B!(y, cg, x)
@test_approx_eq y ye

@test size(cg,1) == size(A,2)
@test size(cg,2) == size(A,2)

######## basic 1-threaded cg ###########
beta = BayesianDataFusion.parallel_cg(cg, x)[1]
beta_e = (A'*A + eye(20)*0.5) \ x
@test_approx_eq beta beta_e
