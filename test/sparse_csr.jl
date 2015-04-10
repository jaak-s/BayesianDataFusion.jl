using BayesianDataFusion
using Base.Test

X = sprand(50, 100, 0.1)
Xr = sparse_csr(X)

@test size(X) == size(Xr)
@test eltype(X) == eltype(Xr)

y1 = rand(100)
y2 = rand(50)

u1 = X * y1
u2 = X' * y2

v1 = Xr * y1
v2a = Ac_mul_B(Xr, y2)
v2b = At_mul_B(Xr, y2)

@test_approx_eq u1 v1
@test_approx_eq u2 v2a
@test_approx_eq u2 v2b
