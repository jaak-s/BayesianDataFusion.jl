using BayesianDataFusion
using Base.Test

addprocs(2)
@everywhere using BayesianDataFusion

W = sprand(50, 10, 0.1)
X = sprand(50, 20, 0.1)

rd = RelationData(W, class_cut = 0.5, feat1 = X)
assignToTest!(rd.relations[1], 2)

result = macau(rd, burnin = 2, psamples = 2, num_latent=5, latent_pids=workers())
@test result["latent_multi_threading"] == true
