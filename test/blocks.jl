using DataFrames
using BayesianDataFusion
using Base.Test

X = IndexedDF(DataFrame(
    A=[2,2,3,3,4,4,5,1,6],
    B=[1,3,1,3,1,3,1,3,1],
    v=[1.0:1:9.0]))

rd = RelationData(X, class_cut = 0.5)

blocks1 = BayesianDataFusion.make_blocks(rd.entities[1])
blocks2 = BayesianDataFusion.make_blocks(rd.entities[2])

@test blocks1[1].ux  == [2,3,4]
@test blocks1[1].vx  == [1,3]
@test blocks1[1].Yma[:,1] == [1.0, 2.0]
@test blocks1[1].Yma[:,2] == [3.0, 4.0]
@test blocks1[1].Yma[:,3] == [5.0, 6.0]

@test size(blocks1[1].Yma) == (2, 3)
@test size(blocks1[2].Yma) == (1, 2)
@test size(blocks1[3].Yma) == (1, 1)

@test size(blocks2[1].Yma) == (5, 1)
@test size(blocks2[2].Yma) == (4, 1)
@test size(blocks2[3].Yma) == (0, 1)

sample_u = rand(5, 6)
sample_m = rand(5, 3)
mu_u     = rand(5)
Lambda_u = eye(5)
alpha    = 2.0

X1 = BayesianDataFusion.sample_users_blocked(blocks1[1], sample_m, alpha, mu_u, Lambda_u)
@test size(X1) == (5, 3)

X2 = BayesianDataFusion.sample_users_blocked(blocks2[3], sample_u, alpha, mu_u, Lambda_u)
@test size(X2) == (5, 1)
