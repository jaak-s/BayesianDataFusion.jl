export pmult, imult
export psparse, ParallelSparseMatrix
export SparseMatrixCSR, sparse_csr

function normsq{T}(x::Vector{T})
  s = zero(T)
  @inbounds @simd for i=1:length(x)
    s += x[i]*x[i]
  end
  s
end

function prod_add!(p, mult, r)
  @inbounds @simd for i=1:length(p)
    p[i] = mult*p[i] + r[i]
  end
end

function add_prod!(x, mult, v)
  @inbounds @simd for i=1:length(x)
    x[i] += mult*v[i]
  end
end

function sub_prod!(x, mult, v)
  @inbounds @simd for i=1:length(x)
    x[i] -= mult*v[i]
  end
end

## K.A -> A
function parallel_cg(x, A, b;
         tol::Real=size(A,2)*eps(), maxiter::Integer=size(A,2))
    tol = tol * norm(b)
    r = b - A * x
    p = copy(r)
    bkden = zero(eltype(x))
    err   = norm(r)

    for iter = 1:maxiter
        err < tol && return x, err, iter
        bknum = normsq(r)

        if iter > 1
            bk = bknum / bkden
            prod_add!(p, bk, r)
        end
        bkden = bknum

        z = A * p

        ak = bknum / dot(z, p)

        add_prod!(x, ak, p)
        sub_prod!(r, ak, z)
        
        err = norm(r)
    end
    x, err, maxiter
end

type ParallelSparseMatrix{TF}
  F::TF
  refs::Vector{RemoteRef}
  procs::Vector{Int}
end

function psparse(F, procs)
  ParallelSparseMatrix(
    F,
    map(i -> @spawnat(i, fetch(F)), procs),
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
                        results[:,idx] = remotecall_fetch(procs[p], mfun, Frefs[p], U[:,idx])
                    end
                end
            end
        end
    end
    results
end
