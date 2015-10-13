using BayesianDataFusion
#using BayesianDataFusion.ParallelMatrix
using Base.Test

using Compat

######### test hilbert sorting #######
rows = [1:4; 1:4; 1:4; 1:4]
cols = [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]

r2, c2 = sort_hilbert(rows, cols)


######### test block order ########
@test BayesianDataFusion.block_order( [1, 2, 5, 4, 3, 1], 2:3 ) == [3, 2]
@test BayesianDataFusion.block_order( [1, 2, 5, 4, 3, 1], 2:4 ) == [3, 4, 2]
@test BayesianDataFusion.block_order( [1, 2, 5, 4, 3, 1], 2:5 ) == [3, 4, 5, 2]
@test BayesianDataFusion.block_order( [1, 2, 5, 4, 3, 1], 2:6 ) == [3, 4, 5, 2, 6]

########## parallel setup ###########
if nprocs() < 3
  addprocs(2)
end
@everywhere using BayesianDataFusion

######### lock test ############
z = SharedArray(UInt32, 16)
@test BayesianDataFusion.sem_init(z) == 0
@test z[1] == 1
@test fetch(@spawnat 2 z[1]) == 1

@test fetch(@spawnat 2 BayesianDataFusion.sem_trywait(z)) == 0
@test z[1] == 0
@test fetch(@spawnat 3 BayesianDataFusion.sem_trywait(z)) == -1
@test z[1] == 0

@test fetch(@spawnat 2 BayesianDataFusion.sem_post(z)) == 0
@test z[1] == 1 

######### parallel mult test #########
rows = Int32[ 1:200; 151:350 ]
cols = Int32[ 151:350; 1:2:399 ]

A = ParallelSBM(rows, cols, workers()[1:2])
sbm = SparseBinMatrix(rows, cols)

@test size(A) == (350, 399)
@test size(sbm) == (350, 399)

y = SharedArray(Float64, size(A, 1))
x = SharedArray(Float64, size(A, 2))

x[1:end] = rand( length(x) )

A_mul_B!(y, A, x)
B = sparse(rows, cols, 1.0)
y_e = B*x
ydirect = A * x

@test_approx_eq y       y_e
@test_approx_eq ydirect y_e

## test error if row and col are different sizes
@test_throws DimensionMismatch SparseBinMatrix( [rows; one(Int32)], cols)
@test_throws DimensionMismatch ParallelSBM(     [rows; one(Int32)], cols, workers()[1:2])

## measuring time
ctimes = BayesianDataFusion.A_mul_B!_time(y, A, x, 3)
@test size(ctimes) == (2, 3)

## At_mul_B test
x9  = rand(A.m)
y9  = At_mul_B(A, x9)
y9e = At_mul_B(B, x9)
@test_approx_eq y9 y9e
@test_approx_eq At_mul_B(sbm, x9) y9e

## testing AtA_mul_B!
xn = SharedArray(Float64, size(A, 2))
AtA_mul_B!(xn, A, x, 0.1)
xn_true = B' * B * x + 0.1 * x
@test_approx_eq xn xn_true

## testing AtA_mul_B! for dense matrix
Bxn = zeros(Float64, size(A,2))
AtA_mul_B!(Bxn, B, x, 0.1)
@test_approx_eq Bxn xn_true

## AtA_mul_B! for SparseBinMatrix
sxn = zeros(Float64, size(sbm,2))
AtA_mul_B!(sxn, sbm, x, 0.1)
@test_approx_eq sxn xn_true

## making sure SparseBinMatrix with Int64 input works
sbm64 = SparseBinMatrix(convert(Vector{Int64}, rows), convert(Vector{Int64}, cols))
@test sbm.rows == sbm64.rows
@test sbm.cols == sbm64.cols

######## make balanced parallel matrix ########
Abal = balanced_parallelsbm(rows, cols, workers()[1:2])
ctimes = BayesianDataFusion.A_mul_B!_time(y, Abal, x, 3)


########     ParallelSBM with CG    ########
#cg   = BayesianDataFusion.CG(A, 0.5, workers()[1:2])
#beta = BayesianDataFusion.parallel_cg(cg, x)[1]
beta = BayesianDataFusion.cg_AtA(A, x, 0.5)
beta_true = (B'*B + eye(size(A,2))*0.5) \ x
@test_approx_eq beta beta_true


######## testing nonshared #########
logic_ns = fetch(@spawnat A.logic[1].where BayesianDataFusion.nonshared(fetch(A.logic[1])) )
@test ! isdefined(logic_ns, :tmp)
@test ! isdefined(logic_ns, :sems)

A_ns = BayesianDataFusion.nonshared(A)
@test size(A) == size(A_ns)
@test ! isdefined(A_ns, :tmp)
@test ! isdefined(A_ns, :sems)
