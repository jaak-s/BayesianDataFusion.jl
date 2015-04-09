using BayesianDataFusion
using Base.Test

X = sprand(50, 100, 0.1)
Y = rand(50, 3)

Z1 = X' * Y
addprocs(2)
@everywhere using BayesianDataFusion

fp = psparse(X, workers())
Z2 = At_mul_B(fp, Y)

@test_approx_eq Z1 Z2

rmprocs( workers() )
