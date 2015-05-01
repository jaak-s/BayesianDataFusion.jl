using BayesianDataFusion
#using BayesianDataFusion.ParallelMatrix
using Base.Test



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
z = SharedArray(Uint32, 16)
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

A = ParallelSBM(rows, cols, workers())

@test size(A) == (350, 399)

y = SharedArray(Float64, size(A, 1))
x = SharedArray(Float64, size(A, 2))

x[1:end] = rand( length(x) )

A_mul_B!(y, A, x)
B = sparse(rows, cols, 1.0)

@test_approx_eq y B*x

## measuring time
ctimes = BayesianDataFusion.A_mul_B!_time(y, A, x, 3)
@test size(ctimes) == (2, 3)

## testing AtA_prod
xn = SharedArray(Float64, size(A, 2))
AtA_mul_B!(xn, A, x, 0.1)
xn_true = B' * B * x + 0.1 * x

@test_approx_eq xn xn_true


######## make balanced parallel matrix ########

Abal = balanced_parallelsbm(rows, cols, workers())
ctimes = BayesianDataFusion.A_mul_B!_time(y, Abal, x, 3)
