using BayesianDataFusion
using Base.Test

addprocs(2)
@everywhere using BayesianDataFusion

rows = Int32[ 1:200; 151:350 ]
cols = Int32[ 151:350; 1:2:399 ]

A = ParallelSBM(rows, cols, workers()[1:2])


########        copyto test         ########
addprocs(2)
@everywhere using BayesianDataFusion
A2 = BayesianDataFusion.copyto(BayesianDataFusion.nonshared(A), workers()[end-1:end])
@test A.m == A2.m
@test A.n == A2.n
@test A.numblocks == A2.numblocks
@test A2.pids     == workers()[end-1:end]
@test length(A.tmp)   == length(A2.tmp)
@test length(A.sems)  == length(A2.sems)
@test length(A.sbms)  == length(A2.sbms)
@test length(A.logic) == length(A2.logic)

########    solving cg after copy   #######
x    = rand(size(A2,2))
beta = BayesianDataFusion.cg_AtA(A2, x, 0.5)
Asp  = sparse(rows, cols, 1.0)
AA   = full(Asp' * Asp)
beta_e = (AA + eye(length(x))*0.5) \ x
@test_approx_eq beta beta_e


########     remote cg     #########
x    = rand(size(A,2))
Ans  = BayesianDataFusion.nonshared(A)
Fref = @spawnat 2 BayesianDataFusion.copyto(Ans, Int[3,4])
beta2 = fetch(@spawnat 2 BayesianDataFusion.cg_AtA_ref(Fref, x, 0.75, 1e-6, length(x)))
Asp   = sparse(rows, cols, 1.0)
AA    = full(Asp' * Asp)
beta2_e = (AA + eye(length(x))*0.75) \ x
@test_approx_eq beta2 beta2_e

########    solve_cg2 test     ########
rhs = rand(size(A,2), 3)
Y   = BayesianDataFusion.solve_cg2(RemoteRef[Fref], rhs, 0.5)
Ye  = (AA + eye(size(rhs,1))*0.5) \ rhs
@test_approx_eq Y Ye


########  Frefs_mul_B on SparseMatrixCSC  #########
Xm = rand(size(Asp,2), 5)
Asp_ref1 = @spawnat 2 fetch(Asp)
Asp_ref2 = @spawnat 3 fetch(Asp)
Ym  = BayesianDataFusion.Frefs_mul_B(RemoteRef[Asp_ref1, Asp_ref2], Xm)
Yme = Asp * Xm
@test_approx_eq Ym Yme

########  Frefs_mul_B on ParallelSBM  #########
Ym_psbm = BayesianDataFusion.Frefs_mul_B(RemoteRef[Fref], Xm)
@test_approx_eq Ym_psbm Yme

########  Frefs_t_mul_B on SparseMatrixCSC  #########
Xt  = rand(size(Asp,1), 5)
Yt  = BayesianDataFusion.Frefs_t_mul_B(RemoteRef[Fref], Xt)
Yte = At_mul_B(Asp, Xt)
@test_approx_eq Yt Yte

########  macau with several SparseMatrixCSC  #######
Y    = sprand(20, 10, 0.2)
feat = sprand(20, 5, 0.5)
rd   = RelationData(Y, class_cut = 0.5, feat1 = feat)
assignToTest!(rd.relations[1], 2)

res = macau(rd, burnin = 2, psamples = 2, verbose = false, compute_ff_size=0, cg_pids=Int[2,3,4], num_latent=5)

########  macau with ParallelSBM  ########
Y     = sprand(20, 10, 0.4)
rows, cols, vals = findnz(Y)
A     = ParallelSBM(rows, cols, workers()[1:2])
rd2   = RelationData(Y, class_cut = 0.5, feat1 = A)
assignToTest!(rd.relations[1], 2)

res2  = macau(rd2, burnin = 2, psamples = 2, verbose = false, compute_ff_size=0, cg_pids=Vector{Int}[ [2, 2,3], [4, 4,5] ], num_latent=5)
