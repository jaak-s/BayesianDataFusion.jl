using BayesianDataFusion
using Base.Test

X = sprand(50, 100, 0.1)
Xr = sparse_csr(X)

@test size(X) == size(Xr)
@test size(X,1) == size(Xr,1)
@test size(X,2) == size(Xr,2)
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

@test_approx_eq full(At_mul_B(X,X)) full(At_mul_B(Xr,Xr))

rows = [1, 2, 2, 4]
cols = [2, 1, 3, 3]
vals = [0.1, 0.2, 0.15, 0.3]
z    = rand(3)
w1   = sparse(rows, cols, vals) * z
w2   = sparse_csr(rows, cols, vals) * z
@test_approx_eq w1 w2
