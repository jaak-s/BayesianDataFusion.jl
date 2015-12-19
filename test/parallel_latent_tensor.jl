using BayesianDataFusion
using Base.Test

using DataFrames

if nprocs() <= 1
  addprocs(2)
  @everywhere using BayesianDataFusion
end

A = randn(15,2);
B = randn(20,2);
C = randn(2,2);

X = Float64[ sum(A[i,:].*B[j,:].*C[k,:]) for i in 1:size(A,1), j in 1:size(B,1), k in 1:size(C,1)]

df = DataFrame(A=Int64[], B=Int64[], C=Int64[], v=Float64[])
for i=1:size(A,1), j=1:size(B,1), k=1:size(C,1)
  push!(df, Any[i, j, k, X[i,j,k]])
end

rd = RelationData(df)
@test rd.entities[1].name == "A"
@test rd.entities[2].name == "B"
@test rd.entities[3].name == "C"

assignToTest!(rd.relations[1], 10)
result = macau(rd, burnin=50, psamples=10, num_latent=2, verbose=true, latent_pids=workers())
@test result["latent_multi_threading"] == true

## checking prediction for whole matrix for tensors
Yhat = pred_all(rd.relations[1])
@test size(Yhat) == size(X)
yprod   = rd.entities[1].model.sample[:,4]
yprod .*= rd.entities[2].model.sample[:,2]
yprod .*= rd.entities[3].model.sample[:,1]
@test_approx_eq Yhat[4,2,1] sum(yprod)+rd.relations[1].model.mean_value
