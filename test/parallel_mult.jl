using BayesianDataFusion
using Base.Test

X = sprand(50, 100, 0.1)
Y1 = rand(50, 3)
Y2 = rand(100, 2)

Z1 = X' * Y1
Z2 = X  * Y2
addprocs(2)
@everywhere using BayesianDataFusion

fp  = psparse(X, workers())
Z1p = At_mul_B(fp, Y1)
Z2p = fp * Y2

@test_approx_eq Z1 Z1p
@test_approx_eq Z2 Z2p

W  = sprand(50, 10, 0.1)
rd = RelationData(W, class_cut = 0.5, feat1 = fp)
assignToTest!(rd.relations[1], 2)

result = macau(rd, burnin = 2, psamples = 2, num_latent=5)

@test size(rd.entities[1].model.beta) == (size(fp,2), 5)

rmprocs( workers() )
