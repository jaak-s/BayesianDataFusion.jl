using BayesianDataFusion
using Base.Test

if nprocs() <= 1
  addprocs(2)
  @everywhere using BayesianDataFusion
end

#W = sprand(50, 10, 0.1)
X = sprand(50, 20, 0.1)

A = randn(50, 2);
B = randn(10, 2);

W = sparse(A * B')

rd = RelationData(W, class_cut = 0.5, feat1 = X)
assignToTest!(rd.relations[1], 50)

#result = macau(rd, burnin = 10, psamples = 10, num_latent=5, latent_pids=workers())
#@test result["latent_multi_threading"] == true

result0 = macau(rd, burnin = 10, psamples = 10, num_latent=5, latent_pids=Int[])
@test result0["latent_multi_threading"] == false
