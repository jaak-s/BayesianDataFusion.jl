export pmult, imult
export psparse, ParallelSparseMatrix
export SparseMatrixCSR, sparse_csr

type ParallelSparseMatrix{TF}
  F::TF
  refs::Vector{Any}
  procs::Vector{Int}
end

function psparse(F, procs)
  ParallelSparseMatrix(
    F,
    Any[@spawnat(i, fetch(F)) for i in procs],
    procs)
end

import Base.At_mul_B
import Base.isempty
import Base.size
import Base.eltype
import Base.Ac_mul_B

At_mul_B(A::ParallelSparseMatrix, U::AbstractMatrix) = pmult(size(A.F,2), A.refs, U, A.procs, imult)
*(A::ParallelSparseMatrix, U::AbstractMatrix)        = pmult(size(A.F,1), A.refs, U, A.procs, mult)
isempty(A::ParallelSparseMatrix) = isempty(A.F)
size(A::ParallelSparseMatrix)    = size(A.F)
size(A::ParallelSparseMatrix, i::Int) = size(A.F, i::Int)
eltype(A::ParallelSparseMatrix)       = eltype(A.F)
Ac_mul_B(A::ParallelSparseMatrix, B::ParallelSparseMatrix) = Ac_mul_B(A.F, B.F)
At_mul_B(A::ParallelSparseMatrix, B::ParallelSparseMatrix) = At_mul_B(A.F, B.F)


###### CSR matrix ######

type SparseMatrixCSR{Tv,Ti}
  csc::SparseMatrixCSC{Tv,Ti}
end

sparse_csr(csc::SparseMatrixCSC) = SparseMatrixCSR(csc')
sparse_csr(rows, cols, vals) = SparseMatrixCSR(sparse(cols, rows, vals))

At_mul_B(A::SparseMatrixCSR, u::AbstractVector) = A.csc * u
Ac_mul_B(A::SparseMatrixCSR, u::AbstractVector) = A.csc * u
*(A::SparseMatrixCSR, u::AbstractVector) = At_mul_B(A.csc, u)
At_mul_B(A::SparseMatrixCSR, B::SparseMatrixCSR) = A_mul_Bt(A.csc, B.csc)
isempty(A::SparseMatrixCSR) = isempty(A.csc)

eltype(A::SparseMatrixCSR) = eltype(A.csc)
function size(A::SparseMatrixCSR)
  m,n = size(A.csc)
  return n,m
end
size(A::SparseMatrixCSR,d) = (d>2 ? 1 : size(A)[d])


###### parallel multiplication ######

function imult(Fref, u)
  return At_mul_B(fetch(Fref), u)
end

function mult(Fref, u)
  #Flocal = fetch(Fref)
  #if size(Flocal,2) != size(u,1)
  #  error(@sprintf("#columns of F(%d) has to equal number of rows of U(%d).", size(Flocal,2), size(u,1)) )
  #end
  return fetch(Fref) * u
end

## setup:
## feat = genes.F
## Frefs = map(i -> @spawnat(i, fetch(feat)), workers())
function pmult(nrows::Integer, Frefs, U, procs, mfun)
    np = length(procs)  # determine the number of processes available
    n  = size(U,2)
    results = zeros(nrows, n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p in 1:length(procs)
            if procs[p] != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[:,idx] = remotecall_fetch(mfun, procs[p], Frefs[p], U[:,idx])
                    end
                end
            end
        end
    end
    results
end
