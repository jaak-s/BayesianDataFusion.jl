using BayesianDataFusion
using Base.Test

using DataFrames

# IndexedDF
X = IndexedDF(DataFrame(A=[2,2,3], B=[1,3,4], C=[0., -1., 0.5]), [4,4])
@test size(getData(X, 1, 1)) == (0,3)
@test size(getData(X, 1, 4)) == (0,3)

x12 = getData(X, 1, 2)
@test size(x12) == (2,3)
@test x12[:,1] == [2,2]
@test x12[:,2] == [1,3]
@test x12[:,3] == [0., -1.]

@test size(getData(X, 2, 2)) == (0,3)

# IndexedDF with tuple input
X2 = IndexedDF(DataFrame(A=[2,2,3], B=[1,3,4], C=[0., -1., 0.5]), (4,4))
