using BayesianDataFusion
using Base.Test

using Compat

rows = [1:3; 2:4; 1:4]
cols = [1,1,1, 2,2,2, 3,3,3,3]
m = SparseBinMatrix(rows, cols)
@test size(m) == (4,3)

m2 = m[ [true, false, true, false], : ]
@test size(m2) == (2,3)
@test length(m2.rows) == 5
rows2 = [1:2; 2; 1:2]
cols2 = [1,1, 2, 3,3]
@test m2.rows == rows2
@test m2.cols == cols2
