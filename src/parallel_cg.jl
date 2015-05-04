######## CG ########

## CG is stored at cg thread
## the system has following threads:
## 1) main thread (macau control)
## 2) cg running thread
## 3) matrix multiply threads (pids, stored at F)
type CG{FT}
  n::Int
  F::FT
  lambda::Float64
  ## vectors used by parallel_cg code
  sh1::SharedVector{Float64}
  sh2::SharedVector{Float64}
end

CG(F, lambda, pids::Vector{Int}) = CG(
  size(F,2),
  F,
  lambda, 
  SharedArray(Float64, size(F,2), pids=pids),
  SharedArray(Float64, size(F,2), pids=pids)
)

import Base.size
size(cg::CG) = (cg.n, cg.n)
size(cg::CG, d::Int) = d <= 2 ? cg.n : 1

import Base.A_mul_B!
function A_mul_B!{FT}(y::AbstractVector{Float64}, rcg::CG{FT}, x::AbstractVector{Float64})
  AtA_mul_B!(y, rcg.F, x, rcg.lambda)
end

## computes y = (F'F + lambda \eye) x
function AtA_mul_B!(y::AbstractVector{Float64}, F::AbstractMatrix, x::AbstractVector{Float64}, lambda::Float64)
  length(y) == length(x) || throw(DimensionMismatch("length(y)=$(length(y)) must equal length(x)=$(length(x))"))
  y[1:end] = F'*F*x + lambda*x
  return nothing
end

## called from main thread
function solve_remote(cgref::RemoteRef, rhs::Vector{Float64}, lambda::Float64; tol=length(rhs)*eps(), maxiter=length(rhs))
  remotecall_fetch(cgref.where, solve_ref, cgref, rhs, lambda, tol, maxiter)
end

function solve_ref(cgref::RemoteRef, rhs::Vector{Float64}, lambda::Float64, tol=length(rhs)*eps(), maxiter=length(rhs))
  cg = fetch(cgref)::CG
  cg.lambda = lambda
  parallel_cg(cg, rhs, tol=tol, maxiter=maxiter)
end


######### cg code ##########
function normsq{T}(x::Vector{T})
  s = zero(T)
  @inbounds @simd for i=1:length(x)
    s += x[i]*x[i]
  end
  s
end

norm2{T}(x::Vector{T}) = sqrt(normsq(x))

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
function parallel_cg(A::CG, b::AbstractVector{Float64};
         tol::Real=size(A,2)*eps(), maxiter::Int=size(A,2))
    tol = tol * norm(b)
    ## x is set initially to 0
    x = zeros(Float64, length(b))
    #r = b - A * x
    r = zeros(length(x)) + b
    ## p and z are parallelized
    p = A.sh1
    z = A.sh2
    Base.copy!(p, r)
    bkden = zero(eltype(x))

    for iter = 1:maxiter
        bknum = normsq(r)::Float64
        err   = sqrt(bknum)

        err < tol && return x, err, iter

        if iter > 1
            bk = bknum / bkden
            prod_add!(p, bk, r)
        end
        bkden = bknum

        ## z = A * p
        A_mul_B!(z, A, p)

        ak = bknum / dot(z, p)

        add_prod!(x, ak, p)
        sub_prod!(r, ak, z)
    end
    x, sqrt(normsq(r)), maxiter
end
