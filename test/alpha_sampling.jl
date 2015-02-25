using BayesianDataFusion
using Base.Test

using DataFrames

Y  = sprand(15,10, 0.1)
rd = RelationData(Y, class_cut = 0.5, alpha_sample=true)
assignToTest!(rd.relations[1], 2)

result = macau(rd, burnin = 5, psamples = 6, verbose = false)
@test rd.relations[1].model.alpha > 0
