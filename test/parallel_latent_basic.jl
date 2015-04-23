using BayesianDataFusion
using Base.Test

X = sprand(50, 20, 0.1)

A = randn(50, 2);
B = randn(10, 2);

W = sparse(A * B')

rd = RelationData(W, class_cut = 0.5, feat1 = X)
assignToTest!(rd.relations[1], 50)

result = macau(rd, burnin = 1, psamples = 1, num_latent=5, verbose=false)

fdata = FastIDF(rd.relations[1].data)

### checking if sampling is same as single-threaded

## testing sample_user_basic
e1 = 1
e2 = 2
for i = 1:size(A,1)
  srand(i)
  s1 = BayesianDataFusion.sample_user2(rd.entities[e1], i, rd.entities[e1].model.mu, Int[e1], Vector{Int}[ Int[e2]])

  srand(i)
  s2 = BayesianDataFusion.sample_user_basic(i, fdata, e1, rd.relations[1].model.mean_value, rd.entities[e2].model.sample', 5.0, rd.entities[e1].model.mu, rd.entities[e1].model.Lambda)
  @test_approx_eq s1 s2
end
