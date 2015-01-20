using IterativeSolvers

type RidgeReg <: AbstractArray{Float64, 2}
    X
    lambda::Float64
end

import Base.size
import Base.*
import Base.issym

function *(rr::RidgeReg, v::Vector{Float64})
    rr.X' * (rr.X * v) + rr.lambda*v
end

function ridge_solve(X, rhs, lambda)
    rr1 = RidgeReg(X, lambda)
    return cg(rr1, rhs)[1]
end

size(rr::RidgeReg)  = ( size(rr.X, 2), size(rr.X, 2) )
issym(rr::RidgeReg) = true
