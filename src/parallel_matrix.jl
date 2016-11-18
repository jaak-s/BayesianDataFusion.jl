#module ParallelMatrix

export SparseBinMatrix, ParallelLogic, ParallelSBM, balanced_parallelsbm
export sort_hilbert
export AtA_mul_B!

using Compat

type SparseBinMatrix
  m::Int64
  n::Int64
  mrange::UnitRange{Int32}
  nrange::UnitRange{Int32}
  ## data
  rows::Vector{Int32}
  cols::Vector{Int32}
end

function SparseBinMatrix(m, n, rows::Vector{Int32}, cols::Vector{Int32})
  length(rows) == length(cols) || throw(DimensionMismatch("length(rows) must equal length(cols)"))
  return SparseBinMatrix(m, n, minimum(rows):maximum(rows), minimum(cols):maximum(cols), rows, cols)
end
SparseBinMatrix(rows::Vector{Int32}, cols::Vector{Int32}) = SparseBinMatrix(maximum(rows), maximum(cols), rows, cols)
SparseBinMatrix(rows::Vector{Int64}, cols::Vector{Int64}) = SparseBinMatrix(convert(Vector{Int32}, rows), convert(Vector{Int32}, cols))

import Base.getindex
function getindex(sbm::SparseBinMatrix, rows::Vector{Bool})
  length(rows) == sbm.m || throw(DimensionMismatch("length(rows) must equal size(sbm,1)"))
  idx = rows[sbm.rows]
  colidx = sbm.cols[idx]
  rsum   = cumsum(rows)
  rowidx = rsum[ sbm.rows[idx] ]
  out = SparseBinMatrix(sum(rows), sbm.n, one(Int32):convert(Int32,sum(rows)), sbm.nrange, rowidx, colidx)
  return out
end

function getindex(sbm::SparseBinMatrix, rows::Vector{Bool}, cols::Colon)
  return getindex(sbm, rows)
end

function getindex(sbm::SparseBinMatrix, rows::Vector{Bool}, cols::UnitRange{Int64})
  return getindex(sbm, rows)
end

import Base.ndims
ndims(sbm::SparseBinMatrix) = 2

type ParallelLogic
  ## for parallel compute
  mblocks::Vector{UnitRange{Int32}}
  nblocks::Vector{UnitRange{Int32}}
  mblock_order::Vector{Int64}
  nblock_order::Vector{Int64}

  ## local vectors
  localm::Vector{Float64}
  localn::Vector{Float64}

  ## semaphores
  tmp::SharedVector{Float64}
  sems::Vector{SharedArray{UInt32,1}}

  ## constructor for non-shared elements
  ParallelLogic(mblocks, nblocks, mblock_order, nblock_order, localm, localn) = new(mblocks, nblocks, mblock_order, nblock_order, localm, localn)
  ParallelLogic(mblocks, nblocks, mblock_order, nblock_order, localm, localn, tmp, sems) = new(mblocks, nblocks, mblock_order, nblock_order, localm, localn, tmp, sems)
end

nonshared(A::ParallelLogic) = ParallelLogic(A.mblocks, A.nblocks, A.mblock_order, A.nblock_order, A.localm, A.localn)

type ParallelSBM
  m::Int64
  n::Int64
  pids::Vector{Int64}
  sbms::Vector{Future}  ## SparseBinMatrix
  logic::Vector{Future} ## ParallelLogic

  numblocks::Int                ## number of blocks, each has semaphore
  tmp::SharedVector{Float64}    ## for storing middle vector in A'A 
  sh1::SharedVector{Float64}    ## length(sh1) = A.n
  sh2::SharedVector{Float64}    ## length(sh2) = A.n
  sems::Vector{SharedArray{UInt32,1}} ## semaphores

  ## constructors
  ParallelSBM(m, n, pids, sbms, logic, numblocks) = new(m, n, pids, sbms, logic, numblocks)
  ParallelSBM(m, n, pids, sbms, logic, numblocks, tmp, sh1, sh2, sems) = new(m, n, pids, sbms, logic, numblocks, tmp, sh1, sh2, sems)
end

nonshared(A::ParallelSBM) = ParallelSBM(A.m, A.n, A.pids, A.sbms, A.logic, A.numblocks)

function make_sems(numblocks::Int, pids::Vector{Int})
  sems = SharedArray{UInt32,1}[SharedArray(UInt32, 16, pids=pids) for i=1:numblocks]
  for sem in sems
    sem_init(sem)
  end
  return sems
end

function block_order(counts, blocks)
  bcounts = counts[blocks]
  return blocks[sortperm(bcounts, rev=true)]
end

function ParallelSBM(rows::Vector{Int}, cols::Vector{Int}, pids::Vector{Int}, weights=ones(length(pids)), m=convert(Int32, maximum(rows)), n=convert(Int32, maximum(cols)), numblocks=length(pids)*2)
  return ParallelSBM(convert(Vector{Int32}, rows), convert(Vector{Int32}, cols), pids, weights=weights, m=m, n=n, numblocks=numblocks)
end

function ParallelSBM(rows::Vector{Int32}, cols::Vector{Int32}, pids::Vector{Int}=Int[]; weights=ones(length(pids)), m=maximum(rows), n=maximum(cols), numblocks=length(pids)*2 )
  length(rows) == length(cols) || throw(DimensionMismatch("length(rows) must equal length(cols)"))

  shtmp = SharedArray(Float64, convert(Int, m), pids=pids)
  sh1   = SharedArray(Float64, convert(Int, n), pids=pids)
  sh2   = SharedArray(Float64, convert(Int, n), pids=pids)
  sems  = make_sems(numblocks, pids)
  ps = ParallelSBM(m, n, pids, Future[], Future[], numblocks, shtmp, sh1, sh2, sems)
  ranges  = make_lengths(length(rows), weights)
  mblocks = make_blocks(m, convert(Int32, numblocks) )
  nblocks = make_blocks(n, convert(Int32, numblocks) )
  mblock_grid = zeros(Int, numblocks, length(pids))
  nblock_grid = zeros(Int, numblocks, length(pids))
  for i in 1:length(pids)
    sbm = SparseBinMatrix(m, n, rows[ranges[i]], cols[ranges[i]])
    sbm_ref = @spawnat pids[i] fetch(sbm)
    push!(ps.sbms, sbm_ref)
    mblock_grid[:,i] = [! isempty(intersect(sbm.mrange, i)) for i in mblocks]
    nblock_grid[:,i] = [! isempty(intersect(sbm.nrange, i)) for i in nblocks]
  end
  mblock_counts = vec(sum(mblock_grid, 2))
  nblock_counts = vec(sum(nblock_grid, 2))
  for i in 1:length(pids)
    mb = block_order(mblock_counts, find(mblock_grid[:,i]))
    nb = block_order(nblock_counts, find(nblock_grid[:,i]))
    pl_ref = @spawnat pids[i] ParallelLogic(mblocks, nblocks, mb, nb, zeros(m), zeros(n), ps.tmp, ps.sems )
    push!(ps.logic, pl_ref)
  end
  return ps
end

## copies ParallelSBM to new pids
function copyto(F::ParallelSBM, pids::Vector{Int})
  length(pids) == length(F.pids) || throw(DimensionMismatch("length(pids)=$(length(pids)) must equal length(F.pids)=$(length(F.pids))"))

  shtmp = SharedArray(Float64, size(F, 1), pids=pids)
  sh1   = SharedArray(Float64, size(F, 2), pids=pids)
  sh2   = SharedArray(Float64, size(F, 2), pids=pids)
  sems  = make_sems(F.numblocks, pids)
  ps    = ParallelSBM(F.m, F.n, pids, Future[], Future[], F.numblocks, shtmp, sh1, sh2, sems)

  for i in 1:length(F.sbms)
    push!(ps.sbms, @spawnat(pids[i], fetch(F.sbms[i])) )

    l = fetch(@spawnat F.pids[i] nonshared(fetch(F.logic[i])))
    logic_ref = @spawnat pids[i] ParallelLogic(l.mblocks, l.nblocks, l.mblock_order, l.nblock_order, zeros(F.m), zeros(F.n), ps.tmp, ps.sems)
    push!(ps.logic, logic_ref)
  end
  return ps
end


sem_init(x::SharedArray)    = ccall(:sem_init, Cint, (Ptr{Void}, Cint, Cuint), x, 1, one(UInt32))
sem_wait(x::SharedArray)    = ccall(:sem_wait, Cint, (Ptr{Void},), x)
sem_trywait(x::SharedArray) = ccall(:sem_trywait, Cint, (Ptr{Void},), x)
sem_post(x::SharedArray)    = ccall(:sem_post, Cint, (Ptr{Void},), x)

gmean(x) = prod(x) ^ (1 / length(x))
pretty(x) = "[" * join([@sprintf("%.3f", i) for i in x], ", ") * "]"

function balanced_parallelsbm(rows::Vector{Int32}, cols::Vector{Int32}, pids::Vector{Int}; numblocks=length(pids)*2, niter=4, ntotal=30, keeplast=4, verbose=false)
  weights = ones(length(pids))
  y = SharedArray(Float64, convert(Int, maximum(rows)) )
  x = SharedArray(Float64, convert(Int, maximum(cols)) )
  local psbm
  for i = 1:niter
    psbm   = ParallelSBM(rows, cols, pids, numblocks=numblocks, weights=weights)
    times  = A_mul_B!_time(y, psbm, x, ntotal)
    ctimes = vec(mean(times[:,end-keeplast:end], 2))
    meantime  = gmean(ctimes)
    weights .*= (meantime ./ ctimes) .^ (1/(1+0.2*i))
    weights   = weights ./ sum(weights)
    verbose && println("$i. ctimes  = ", pretty(ctimes) )
    verbose && println("$i. weights = ", pretty(weights) )
  end
  return psbm
end

function make_blocks(n::Int32, nblocks::Int32)
  bsize   = 8 * ceil(n / nblocks / 8)
  if (bsize-1) * nblocks > n
    bsize = ceil(n / nblocks)
  end
  return  UnitRange{Int32}[convert(Int32, 1+(i-1)*bsize) : convert(Int32, min(n, i*bsize))
                           for i in 1:nblocks]
end

function make_lengths(total::Int, weights)
    wnorm  = weights ./ sum(weights)
    n      = convert(Vector{Int64}, round(wnorm * total))
    excess = sum(n) - total
    i = 0
    while excess != 0
        n[i+1] -= sign(excess)
        excess  = sum(n) - total
        i = (i + 1) % length(n)
    end
    ranges = UnitRange{Int}[]
    k = 1
    for len in n
      push!(ranges, k:(k+len-1))
      k += len
    end
    return ranges
end

## waits 100x to see that x[n] == value
function busywait(x::SharedArray{Int,1}, n::Int, value::Int, ntimes=100)
  for i=1:ntimes
    if x[n] != value
      return false
    end
  end
  return true
end

import Base.size
import Base.isempty
size(X::SparseBinMatrix) = (X.m, X.n)
size(X::SparseBinMatrix, d::Int) = d==1 ? X.m : X.n
isempty(X::SparseBinMatrix)      = X.m == 0 || X.n == 0

size(X::ParallelSBM) = (X.m, X.n)
size(X::ParallelSBM, d::Int) = d==1 ? X.m : X.n
isempty(X::ParallelSBM)      = X.m == 0 || X.n == 0

import Base.A_mul_B!
import Base.At_mul_B!
import Base.At_mul_B
import Base.*

## multiplication: y = A * x
*{Tx}(A::SparseBinMatrix, x::AbstractArray{Tx,1}) = (y = zeros(Tx, A.m); A_mul_B!(y, A, x); y)
*{Tx}(A::ParallelSBM,     x::AbstractArray{Tx,1}) = (y = zeros(Tx, A.m); A_mul_B!(y, A, x); y)

function A_mul_B!{Tx}(y::AbstractArray{Tx,1}, A::SparseBinMatrix, x::AbstractArray{Tx,1})
    A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
    A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
    fill!(y, zero(Tx) )
    rows = A.rows
    cols = A.cols
    @inbounds for i = 1:length(rows)
        y[rows[i]] += x[cols[i]]
    end
    return
end

## multiplication: y = A' * x
At_mul_B{Tx}(A::SparseBinMatrix, x::AbstractArray{Tx,1}) = (y = zeros(Tx, A.n); At_mul_B!(y, A, x); y)

function At_mul_B!{Tx}(y::AbstractArray{Tx,1}, A::SparseBinMatrix, x::AbstractArray{Tx,1})
    A.n == length(y) || throw(DimensionMismatch("A.n=$(A.n) must equal length(y)=$(length(y))"))
    A.m == length(x) || throw(DimensionMismatch("A.m=$(A.m) must equal length(x)=$(length(x))"))
    fill!(y, zero(Tx) )
    rows = A.rows
    cols = A.cols
    @inbounds for i = 1:length(rows)
        y[cols[i]] += x[rows[i]]
    end
    return
end

function A_mul_B!{Tx}(y::SharedArray{Tx,1}, A::ParallelSBM, x::SharedArray{Tx,1})
  A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
  A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
  y[1:end] = zero(Tx)
  np = length(A.pids)
  @sync begin
    for p in 1:length(A.pids)
      pid = A.pids[p]
      if pid != myid() || np == 1
        @async begin
          remotecall_wait(partmul_ref, pid, y, A.sbms[p], A.logic[p], x)
        end
      end
    end
  end
  ## done
end

function At_mul_B!{Tx}(y::SharedArray{Tx,1}, A::ParallelSBM, x::SharedArray{Tx,1})
  A.m == length(x) || throw(DimensionMismatch("A.m=$(A.m) must equal length(x)=$(length(x))"))
  A.n == length(y) || throw(DimensionMismatch("A.n=$(A.n) must equal length(y)=$(length(y))"))
  y[1:end] = zero(Tx)
  np = length(A.pids)
  @sync begin
    for p in 1:length(A.pids)
      pid = A.pids[p]
      if pid != myid() || np == 1
        @async begin
          remotecall_wait(partmul_t_ref, pid, y, A.sbms[p], A.logic[p], x)
        end
      end
    end
  end
end

import Base.*

function At_mul_B{Tx}(A::ParallelSBM, x::AbstractArray{Tx,1})
  A.m == length(x) || throw(DimensionMismatch("A.m=$(A.m) must equal length(x)=$(length(x))"))
  A.tmp[1:end] = x
  At_mul_B!(A.sh1, A, A.tmp)
  y = zeros(Tx, A.n)
  y[1:end] = A.sh1
  return y
end

function prod_copy!{Tx}(y::SharedArray{Tx,1}, v::Tx, x::SharedArray{Tx,1})
  length(y) == length(x) || throw(DimensionMismatch("length(y)=$(length(y)) must equal length(x)=$(length(x))"))
  @inbounds @simd for i = 1:length(y)
    y[i] = v * x[i]
  end
  return nothing
end

## computes A'A*x + lambda*x
function AtA_mul_B!{Tx}(y::SharedArray{Tx,1}, A::ParallelSBM, x::SharedArray{Tx,1}, lambda::Float64)
  A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
  A.n == length(y) || throw(DimensionMismatch("A.n=$(A.n) must equal length(y)=$(length(y))"))
  tmp = A.tmp
  tmp[1:end] = zero(Tx)
  np = length(A.pids)
  ## doing tmp = A * x
  @sync begin
    for p in 1:length(A.pids)
      pid = A.pids[p]
      if pid != myid() || np == 1
        @async begin
          remotecall_wait(partmul_ref, pid, tmp, A.sbms[p], A.logic[p], x)
        end
      end
    end
    prod_copy!(y, lambda, x)
  end
  ## doing y += A' * tmp
  @sync begin
    for p in 1:length(A.pids)
      pid = A.pids[p]
      if pid != myid() || np == 1
        @async begin
          remotecall_wait(partmul_t_ref, pid, y, A.sbms[p], A.logic[p], tmp)
        end
      end
    end
  end
  return nothing
end

function A_mul_B!_time{Tx}(y::SharedArray{Tx,1}, A::ParallelSBM, x::SharedArray{Tx,1}, ntimes::Int)
  A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
  A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
  ptime = zeros(length(A.pids), ntimes)
  for i = 1:ntimes
    y[1:end] = zero(Tx)
    ## clearing warmup results
    @sync begin
      for p in 1:length(A.pids)
        pid = A.pids[p]
        if pid != myid() || length(A.pids) == 1
          @async begin
            ptime[p, i] += remotecall_fetch(partmul_time, pid, y, A.sbms[p], A.logic[p], x)
          end
        end
      end
    end
  end
  return ptime
end

import Base.copy!
function copy!{Tx}(to::AbstractArray{Tx,1}, from::AbstractArray{Tx,1}, range)
  @inbounds @simd for i in range
    to[i] = from[i]
  end
  return nothing
end

function add!{Tx}(to::AbstractArray{Tx,1}, from::AbstractArray{Tx,1}, range)
  @inbounds @simd for i in range
    to[i] += from[i]
  end
  return nothing
end

function rangefill!{Tx}(x::AbstractArray{Tx,1}, v::Tx, range)
  @inbounds @simd for i in range
    x[i] = v
  end
  return nothing
end

function partmul_time{Tx}(y::SharedArray{Tx,1}, Aref::Future, logicref::Future, x::SharedArray{Tx,1})
  A     = fetch(Aref)::SparseBinMatrix
  logic = fetch(logicref)::ParallelLogic
  tic();
  partmul(y, A, logic, x)
  return toq()
end

function partmul_ref{Tx}(y::SharedArray{Tx,1}, Aref::Future, logicref::Future, x::SharedArray{Tx,1})
  A     = fetch(Aref)::SparseBinMatrix
  logic = fetch(logicref)::ParallelLogic
  partmul(y, A, logic, x)
end

function partmul_t_ref{Tx}(y::SharedArray{Tx,1}, Aref::Future, logicref::Future, x::SharedArray{Tx,1})
  A     = fetch(Aref)::SparseBinMatrix
  logic = fetch(logicref)::ParallelLogic
  partmul_t(y, A, logic, x)
end

## assumes sizes are correct
function partmul{Tx}(y::SharedArray{Tx,1}, A::SparseBinMatrix, logic::ParallelLogic, x::SharedArray{Tx,1})
  ylocal = logic.localm
  xlocal = logic.localn
  rangefill!(ylocal, zero(Tx), A.mrange)
  copy!(xlocal, x, A.nrange)
  ## standard y = A * x
  rows = A.rows
  cols = A.cols
  @inbounds for i = 1:length(rows)
      ylocal[rows[i]] += xlocal[cols[i]]
  end
  ## adding the result to shared array
  addshared!(y, ylocal, logic.sems, logic.mblocks, logic.mblock_order, A.mrange)
  return nothing
end

## assumes sizes are correct
function partmul_t{Tx}(y::SharedArray{Tx,1}, A::SparseBinMatrix, logic::ParallelLogic, x::SharedArray{Tx,1})
  ylocal = logic.localn
  xlocal = logic.localm
  rangefill!(ylocal, zero(Tx), A.nrange)
  copy!(xlocal, x, A.mrange)
  ## standard y = A' * x
  rows = A.cols
  cols = A.rows
  @inbounds for i = 1:length(rows)
      ylocal[rows[i]] += xlocal[cols[i]]
  end
  ## adding the result to shared array
  addshared!(y, ylocal, logic.sems, logic.nblocks, logic.nblock_order, A.nrange)
  return nothing
end

## does y += x
function addshared!{Tx}(y::SharedArray{Tx,1}, x::AbstractArray{Tx,1}, sems, ranges, order, yrange)
  blocks = copy(order)
  nblocks = length(blocks)
  pid = myid()
  while true
    i = findfirst(blocks)
    i <= 0 && return nothing ## done
    for j = i:nblocks
      block = blocks[j]
      block == 0 && continue
      ## try to get a lock
      if sem_trywait( sems[block] ) < 0
        ## didn't get lock
        continue
      end
      ## copying result to shared array
      add!(y, x, intersect(ranges[blocks[j]], yrange) )
      sem_post( sems[block] )
      blocks[j] = 0
      break
    end
  end
  return nothing
end

######## parallel operations on Frefs
function solve_cg2(Frefs::Vector{Future}, rhs::Matrix{Float64}, lambda::Float64; tol=1e-6, maxiter=size(rhs,1))
  beta = zeros(size(rhs,1), size(rhs,2))
  D    = size(rhs,2)
  i    = 1
  # function to produce the next work item from the queue.
  # in this case it's just an index.
  nextidx() = (idx=i; i+=1; idx)
  @sync begin
    for ref in Frefs
      @async begin
        while true
          idx = nextidx()
          idx > D && break
          beta[:,idx] = remotecall_fetch(ref.where, cg_AtA_ref, ref, rhs[:,idx], lambda, tol, maxiter)
        end
      end
    end
  end
  return beta
end

function A_mul_B_ref(Fref::Future, x::AbstractVector{Float64})
  F = fetch(Fref)
  return F * x
end

function At_mul_B_ref(Fref::Future, x::AbstractVector{Float64})
  F = fetch(Fref)
  return At_mul_B(F, x)
end

## computes y = F * x in parallel (along columns of x)
function Frefs_mul_B(Frefs::Vector{Future}, x::Matrix{Float64})
  m, n = fetch( @spawnat Frefs[1].where size(fetch(Frefs[1])) )
  n == size(x, 1) || throw(DimensionMismatch("Frefs.n=$(n) must equal length(x)=$(length(x))"))
  y = zeros(m, size(x,2))
  D = size(x, 2)
  i = 1
  nextidx() = (idx=i; i+=1; idx)
  @sync begin
    for ref in Frefs
      @async begin
        while true
          idx = nextidx()
          idx > D && break
          y[:,idx] = remotecall_fetch(ref.where, A_mul_B_ref, ref, x[:,idx])
        end
      end
    end
  end
  return y
end

## computes y = F' * x in parallel (along columns of x)
function Frefs_t_mul_B(Frefs::Vector{Future}, x::Matrix{Float64})
  m, n = fetch( @spawnat Frefs[1].where size(fetch(Frefs[1])) )
  m == size(x, 1) || throw(DimensionMismatch("Frefs.m=$(m) must equal length(x)=$(length(x))"))
  y = zeros(n, size(x,2))
  D = size(x, 2)
  i = 1
  nextidx() = (idx=i; i+=1; idx)
  @sync begin
    for ref in Frefs
      @async begin
        while true
          idx = nextidx()
          idx > D && break
          y[:,idx] = remotecall_fetch(ref.where, At_mul_B_ref, ref, x[:,idx])
        end
      end
    end
  end
  return y
end

######## Hilbert ordering ########

## convert (x,y) to d
function xy2d(n, x, y)
    local rx::Bool, ry::Bool
    d = 0::Int64
    s = div(n, 2)
    while s > 0
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) $ ry);
        x, y = rot(s, x, y, rx, ry);
        s = div(s, 2)
    end
    return d
end

function rot(n, x, y, rx::Bool, ry::Bool)
    if ry == false
        if rx == true
            x = n - 1 - x
            y = n - 1 - y
        end
        return (y, x)
    end
    return (x, y)
end

function sort_hilbert(rows, cols)
  maxrc = max(maximum(rows), maximum(cols))
  n = 2 ^ round(Int, ceil(log2( maxrc )))
  h = zeros(Int, length(rows))
  @inbounds @simd for i = 1:length(h)
    h[i] = xy2d(n, rows[i] - 1, cols[i] - 1)
  end
  hsorted = sortperm(h)
  return rows[hsorted], cols[hsorted]
end
