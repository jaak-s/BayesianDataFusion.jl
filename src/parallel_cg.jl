## basic copyto function, sets F up on cg pid
copyto(F::Any, pids::Vector{Int}) = F

## computes y = (F'F + lambda \eye) x
function AtA_mul_B!(y::AbstractVector{Float64}, F, x::AbstractVector{Float64}, lambda::Float64)
  length(y) == length(x) || throw(DimensionMismatch("length(y)=$(length(y)) must equal length(x)=$(length(x))"))
  At_mul_B!(y, F, F*x)
  for i = 1:length(y)
    y[i] += lambda*x[i]
  end
  return nothing
end

## for standard objects nonshared does not do anything
nonshared(A) = A

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
