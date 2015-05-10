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

## sets F up on pids and creates CG object
## should be launched at CG controlling process
function setup_cg(F, lambda, pids)
  Floc = copyto(F, pids)
  return CG{typeof(Floc)}(Floc, lambda, pids)
end

## basic copyto function, sets F up on cg pid
copyto(F::Any, pids::Vector{Int}) = F

import Base.size
size(cg::CG) = (cg.n, cg.n)
size(cg::CG, d::Int) = d <= 2 ? cg.n : 1

import Base.A_mul_B!
function A_mul_B!{FT}(y::AbstractVector{Float64}, rcg::CG{FT}, x::AbstractVector{Float64})
  AtA_mul_B!(y, rcg.F, x, rcg.lambda)
end

## computes y = (F'F + lambda \eye) x
function AtA_mul_B!(y::AbstractVector{Float64}, F, x::AbstractVector{Float64}, lambda::Float64)
  length(y) == length(x) || throw(DimensionMismatch("length(y)=$(length(y)) must equal length(x)=$(length(x))"))
  At_mul_B!(y, F, F*x)
  for i = 1:length(y)
    y[i] += lambda*x[i]
  end
  return nothing
end

## computes CG.F*x
cgA_mul_B(A, x::AbstractVector{Float64}) = (y=zeros(Float64, size(A,1)); cgA_mul_B!(y,A,x); y)
cgA_mul_B!(y, cgref::RemoteRef, x)       = cgA_mul_B!(y, fetch(cgref), x)
cgA_mul_B!(y::AbstractVector{Float64}, cg::CG{Any}, x::AbstractVector{Float64}) = A_mul_B!(y, cg.F, x)
function cgA_mul_B!(y::AbstractVector{Float64}, cg::CG{ParallelSBM}, x::AbstractVector{Float64})
  ysh = cg.F.tmp
  copy!(cg.sh1, x)
  A_mul_B!(ysh, cg.F, cg.sh1)
  copy!(y, ysh)
end

## called from main thread
function solve_remote(cgref::RemoteRef, rhs::Vector{Float64}, lambda::Float64; tol=length(rhs)*eps(), maxiter=length(rhs))
  remotecall_fetch(cgref.where, solve_ref, cgref, rhs, lambda, tol, maxiter)
end

function solve_ref(cgref::RemoteRef, rhs::Vector{Float64}, lambda::Float64, tol=length(rhs)*eps(), maxiter=length(rhs))
  cg = fetch(cgref)::CG
  cg.lambda = lambda
  parallel_cg(cg, rhs, tol=tol, maxiter=maxiter)[1]
end

## for standard objects nonshared does not do anything
nonshared(A) = A

function make_remote_cg(A, cg_pid::Int, pids::Vector{Int})
  Ans  = nonshared(A)
  return @spawnat cg_pid make_cg(Ans, pids)
end

## called by make_remote_cg
function make_cg(Ans, pids::Vector{Int})
  Aloc = copyto(Ans, pids)
  return CG(Aloc, 0.5, pids)
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

## function for calling with remote
cg_AtA_ref(Aref::RemoteRef, b::AbstractVector{Float64}, lambda::Float64, tol::Float64, maxiter::Int=length(b)) = cg_AtA(fetch(Aref), b, lambda, tol=tol, maxiter=maxiter)

## p and z are parallelized (SharedArray)
function cg_AtA(A::ParallelSBM, b::AbstractVector{Float64}, lambda::Float64;
         tol::Float64=size(A,2)*eps(), maxiter::Int=size(A,2))
    return cg_AtA(A, b, lambda, A.sh1, A.sh2, tol=tol, maxiter=maxiter)
end

## non-parallel version
function cg_AtA(A, b::AbstractVector{Float64}, lambda::Float64;
         tol::Float64=size(A,2)*eps(), maxiter::Int=size(A,2))
    return cg_AtA(A, b, lambda, zeros(length(b)), zeros(length(b)), tol=tol, maxiter=maxiter)
end

function cg_AtA(A, b::AbstractVector{Float64}, lambda::Float64, p::AbstractVector{Float64}, z::AbstractVector{Float64};
         tol::Float64=size(A,2)*eps(), maxiter::Int=size(A,2))
    tol = tol * norm(b)
    ## x is set initially to 0
    x = zeros(Float64, length(b))
    #r = b - A * x
    r = zeros(length(x)) + b
    Base.copy!(p, r)
    bkden = zero(eltype(x))

    for iter = 1:maxiter
        bknum = normsq(r)::Float64
        err   = sqrt(bknum)

        err < tol && return x#, err, iter

        if iter > 1
            bk = bknum / bkden
            prod_add!(p, bk, r)
        end
        bkden = bknum

        ## z = (A'A + lambda*\eye) * p
        AtA_mul_B!(z, A, p, lambda)

        ak = bknum / dot(z, p)

        add_prod!(x, ak, p)
        sub_prod!(r, ak, z)
    end
    x#, sqrt(normsq(r)), maxiter
end

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
