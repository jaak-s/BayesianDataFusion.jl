using BayesianDataFusion
using Base.Test

using DataFrames

A = randn(15,2);
B = randn(20,2);
C = ones(1,2);

X = Float64[ sum(A[i,:].*B[j,:].*C[k,:]) for i in 1:size(A,1), j in 1:size(B,1), k in 1:size(C,1)]

df = DataFrame(A=Int64[], B=Int64[], C=Int64[], v=Float64[])
for i=1:size(A,1), j=1:size(B,1), k=1:size(C,1)
  push!(df, {i,j,k,X[i,j,k]})
end

rd = RelationData(df)
assignToTest!(rd.relations[1], 10)
result = BMRF(rd, burnin=50, psamples=10, num_latent=2)

####

A2 = randn(15,2);
B2 = randn(20,2);

X2 = Float64[ sum(A2[i,:].*B2[j,:]) for i in 1:size(A2,1), j in 1:size(B2,1)]

df2 = DataFrame(A=Int64[], B=Int64[], v=Float64[])
for i=1:size(A2,1), j=1:size(B2,1)
  push!(df2, {i,j,X2[i,j]})
end

rd2 = RelationData(df2)
assignToTest!(rd2.relations[1], 10)
result = BMRF(rd2, burnin=50, psamples=10, num_latent=2)
