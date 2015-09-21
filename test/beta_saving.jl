using BayesianDataFusion
using Base.Test

using DataFrames

A = randn(20,2);
B = randn(30,2);

X = Float64[ sum(A[i,:].*B[j,:]) for i in 1:size(A,1), j in 1:size(B,1) ]

df = DataFrame(A=Int64[], B=Int64[], v=Float64[])
for i=1:size(A,1), j=1:size(B,1)
  push!(df, Any[i, j, X[i,j]])
end

rd = RelationData(df)
rd.entities[1].F = randn(size(A,1), 3)

assignToTest!(rd.relations[1], 10)

tmpdir = mktempdir()
try
  result = macau(rd, burnin=5, psamples=10, num_latent=2, verbose=false, output_beta=true, output="$tmpdir/macau-betasaving")
  @test isfile("$tmpdir/macau-betasaving-A-01.binary")
  @test isfile("$tmpdir/macau-betasaving-A-01.beta.binary")
  @test isfile("$tmpdir/macau-betasaving-A-02.beta.binary")
  beta_sample1 = read_binary_float32("$tmpdir/macau-betasaving-A-01.beta.binary")
  @test 2 == size(beta_sample1, 2) ## number of latents
  @test 3 == size(beta_sample1, 1) ## number of features
  beta_sample2 = read_binary_float32("$tmpdir/macau-betasaving-A-10.beta.binary")
  beta_last    = convert(Array{Float32}, rd.entities[1].model.beta)
  @test_approx_eq   beta_sample2 beta_last
finally
  rm(tmpdir, recursive = true)
end
