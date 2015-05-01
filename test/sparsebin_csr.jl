using BayesianDataFusion
using Base.Test

rows = Int32[ 1:200; 151:350 ]
cols = Int32[ 151:350; 1:2:399 ]

######## SparseBinMatrixCSR ###########
csr = SparseBinMatrixCSR(rows, cols)

y = zeros(Float64, size(csr, 1))
x = rand(Float64,  size(csr, 2))

A_mul_B!(y, csr, x)

######### SparseBinMatrix  ############
bin = SparseBinMatrix(rows, cols)
y2 = zeros(Float64, size(bin, 1))
A_mul_B!(y2, bin, x)

A = sparse(rows, cols, 1.0)

@test_approx_eq y  A*x
@test_approx_eq y2 A*x

######### Parallel CSR ###########
