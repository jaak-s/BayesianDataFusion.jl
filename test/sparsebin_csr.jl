using BayesianDataFusion
using Base.Test

rows = Int32[ 1:200; 151:350 ]
cols = Int32[ 151:350; 1:2:399 ]

csr = SparseBinMatrixCSR(rows, cols)

y = zeros(Float64, size(csr, 1))
x = rand(Float64,  size(csr, 2))

A_mul_B!(y, csr, x)

###
bin = SparseBinMatrix(rows, cols)
y2 = zeros(Float64, size(bin, 1))
A_mul_B!(y2, bin, x)
