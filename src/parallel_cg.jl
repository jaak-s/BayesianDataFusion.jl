export pmult, imult
export psparse, ParallelSparseMatrix

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
        println(err)
    end
    x, err, maxiter
end

type ParallelSparseMatrix
  F::SparseMatrixCSC
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

At_mul_B(A::ParallelSparseMatrix, U::AbstractMatrix) = pmult(A.F, A.refs, U, A.procs)
isempty(A::ParallelSparseMatrix) = isempty(A.F)
size(A::ParallelSparseMatrix)    = size(A.F)
size(A::ParallelSparseMatrix, i::Int) = size(A.F, i::Int)
eltype(A::ParallelSparseMatrix)       = eltype(A.F)
Ac_mul_B(A::ParallelSparseMatrix, B::ParallelSparseMatrix) = Ac_mul_B(A.F, B.F)

function imult(Fref, u)
  return At_mul_B(fetch(Fref), u)
end

## setup:
## feat = genes.F
## Frefs = map(i -> @spawnat(i, fetch(feat)), workers())
function pmult(F, Frefs, U, procs)
    np = length(procs)  # determine the number of processes available
    n  = size(U,2)
    results = zeros(size(F,2), n)
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
                        results[:,idx] = remotecall_fetch(procs[p], imult, Frefs[p], U[:,idx])
                    end
                end
            end
        end
    end
    results
end
