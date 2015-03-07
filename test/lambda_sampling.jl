using BayesianDataFusion
using Base.Test

using DataFrames

A = randn(30,2);
B = randn(40,2);

X = Float64[ sum(A[i,:].*B[j,:]) for i in 1:size(A,1), j in 1:size(B,1) ]

df = DataFrame(A=Int64[], B=Int64[], v=Float64[])
for i=1:size(A,1), j=1:size(B,1)
  push!(df, Any[i, j, X[i,j]])
end

## adding features
feat = randn(size(A,1), 2)

rd = RelationData(df)
rd.entities[1].F = feat
rd.entities[1].lambda_beta_sample = true

assignToTest!(rd.relations[1], 10)

result = macau(rd, burnin=50, psamples=10, num_latent=2, verbose=false)
