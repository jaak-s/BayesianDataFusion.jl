using BayesianDataFusion
using Base.Test

addprocs(2)
@everywhere using BayesianDataFusion

rows = Int32[ 1:200; 151:350 ]
cols = Int32[ 151:350; 1:2:399 ]

A = ParallelSBM(rows, cols, workers()[1:2])


########        copyto test         ########
addprocs(2)
@everywhere using BayesianDataFusion
A2 = BayesianDataFusion.copyto(BayesianDataFusion.nonshared(A), workers()[end-1:end])
@test A.m == A2.m
@test A.n == A2.n
@test length(A.tmp)   == length(A2.tmp)
@test length(A.sems)  == length(A2.sems)
@test length(A.sbms)  == length(A2.sbms)
@test length(A.logic) == length(A2.logic)
