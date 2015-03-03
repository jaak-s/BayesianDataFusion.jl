using BayesianDataFusion
using Base.Test

using DataFrames

A = randn(30,2);
B = randn(40,2);

X = Float64[ sum(A[i,:].*B[j,:]) for i in 1:size(A,1), j in 1:size(B,1) ]

df = DataFrame(A=Int64[], B=Int64[], v=Float64[])
for i=1:size(A,1), j=1:size(B,1)
  push!(df, {i, j, X[i,j]})
end

## adding features
feat = randn(size(df,1), 2)
beta = [1.0, -1.0]
df[:,end] = df[:,end] + feat * beta

rd = RelationData(df) #, alpha_sample = true)
rd.relations[1].model.alpha_sample = true
rd.relations[1].F = feat

assignToTest!(rd.relations[1], 10)
@test size(rd.relations[1].test_F) == (10,2)

result = macau(rd, burnin=50, psamples=10, num_latent=2, verbose=false)
