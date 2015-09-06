using BayesianDataFusion
using Base.Test

## testing column dot
X1 = rand(5, 10)
X2 = rand(5, 3)
@test_approx_eq BayesianDataFusion.column_dot(X1, X2, 8, 1) dot(X1[:,8], X2[:,1])

Y  = sprand(15,10, 0.2)
rd = RelationData(Y, class_cut = 0.5)
assignToTest!(rd.relations[1], 2)

# running the data
result = macau_hmc(rd, burnin = 10, psamples = 10, verbose = false)
