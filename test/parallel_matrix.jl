using BayesianDataFusion
#using BayesianDataFusion.ParallelMatrix
using Base.Test

######### copy test ############

y = SharedArray(Float64, 1000)
ylocal = rand(1000)
merror = SharedArray(Int, 8)

mutex  = BayesianDataFusion.make_mutex(10)
ranges = [1+(i-1)*100 : i*100 for i in 1:10]

addshared!(y, ylocal, mutex, ranges, [2,3], merror)

@test_approx_eq y[ranges[1]] zeros(100)
@test_approx_eq y[ranges[2]] ylocal[ranges[2]]
@test_approx_eq y[ranges[3]] ylocal[ranges[3]]
@test_approx_eq y[ranges[4]] zeros(100)

@test merror[1] == 0
@test all(mutex .== 0) == true

########## mutex test ###########
ask_for_lock!(mutex, 1, 7)
@test mutex[1] == 7
release_lock!(mutex, 1, 7)
@test mutex[1] == 0

ask_for_lock!(mutex, 2, 7)
@test mutex[9] == 7
@test ask_for_lock!(mutex, 2, 8) == false
@test mutex[9] == 7
release_lock!(mutex, 2, 7)


######### test hilbert sorting #######
rows = [1:4; 1:4; 1:4; 1:4]
cols = [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]

r2, c2 = sort_hilbert(rows, cols)


########## parallel mult test ###########
if nprocs() < 3
  addprocs(2)
end
@everywhere using BayesianDataFusion
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
