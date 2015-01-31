using BayesianDataFusion
using Base.Test

using DataFrames

# IndexedDF
X = IndexedDF(DataFrame(A=[2,2,3], B=[1,3,4], C=[0., -1., 0.5]), [4,4])
@test nnz(X) == 3
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

@test size(X2) == (4,4)

# testing removing rows
X3 = removeSamples(X2, [2])
@test nnz(X3) == 2
@test size(X3) == (4,4)
x12 = getData(X3, 1, 2)
@test size(x12) == (1,3)
@test size(x12,1) == getCount(X3, 1, 2)

# testing RelationData from sparse matrix
Y  = sprand(15,10, 0.1)
rd = RelationData(Y, class_cut = 0.5)
assignToTest!(rd.relations[1], 2)
@test numTest(rd.relations[1]) == 2
@test length(rd.relations[1].test_label) == 2

# running the data
result = BMRF(rd, burnin = 10, psamples = 10, verbose = false)
@test size(result["predictions"],1) == 2
