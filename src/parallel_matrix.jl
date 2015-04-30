#module ParallelMatrix

export SparseBinMatrix, ParallelLogic, ParallelSBM, balanced_parallelsbm
export SparseBinMatrixCSR
export addshared!, ask_for_lock!, release_lock!
export sort_hilbert

type SparseBinMatrix
  m::Int64
  n::Int64
  mrange::UnitRange{Int32}
  nrange::UnitRange{Int32}
  ## data
  rows::Vector{Int32}
  cols::Vector{Int32}
end

SparseBinMatrix(rows::Vector{Int32}, cols::Vector{Int32}) = SparseBinMatrix(maximum(rows), maximum(cols), minimum(rows):maximum(rows), minimum(cols):maximum(cols), rows, cols)
SparseBinMatrix(m, n, rows::Vector{Int32}, cols::Vector{Int32}) = SparseBinMatrix(m, n, minimum(rows):maximum(rows), minimum(cols):maximum(cols), rows, cols)

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
  sems::Vector{SharedArray{Uint32,1}}
end

type ParallelSBM
  m::Int64
  n::Int64
  pids::Vector{Int64}
  sbms::Vector{RemoteRef}  ## SparseBinMatrix
  logic::Vector{RemoteRef} ## ParallelLogic

  tmp::SharedVector{Float64}  ## for storing middle vector in A'A 
  sems::Vector{SharedArray{Uint32,1}} ## semaphores
end

function ParallelSBM(rows::Vector{Int32}, cols::Vector{Int32}, pids::Vector{Int}=Int[]; weights=ones(length(pids)), m=maximum(rows), n=maximum(cols), numblocks=length(pids)*2 )
  length(rows) == length(cols) || throw(DimensionMismatch("length(rows) must equal length(cols)"))

  sems = SharedArray{Uint32,1}[SharedArray(Uint32, 16, pids=pids) for i=1:numblocks]
  for sem in sems
    sem_init(sem)
  end
  shtmp = SharedArray(Float64, convert(Int, m), pids=pids)
  ps = ParallelSBM(m, n, pids, RemoteRef[], RemoteRef[], shtmp, sems)
  ranges  = make_lengths(length(rows), weights)
  mblocks = make_blocks(m, convert(Int32, numblocks) )
  nblocks = make_blocks(n, convert(Int32, numblocks) )
  for i in 1:length(pids)
    sbm = SparseBinMatrix(m, n, rows[ranges[i]], cols[ranges[i]])
    sbm_ref = @spawnat pids[i] fetch(sbm)
    push!(ps.sbms, sbm_ref)
    mb = find([! isempty(intersect(sbm.mrange, i)) for i in mblocks])
    nb = find([! isempty(intersect(sbm.nrange, i)) for i in nblocks])
    pl_ref = @spawnat pids[i] ParallelLogic(mblocks, nblocks, mb, nb, zeros(m), zeros(n), ps.tmp, sems )
    push!(ps.logic, pl_ref)
  end
  return ps
end

sem_init(x::SharedArray)    = ccall(:sem_init, Cint, (Ptr{Void}, Cint, Cuint), x, 1, one(Uint32))
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
    #ctimes = vec(median(times, 2))
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
size(X::SparseBinMatrix) = (X.m, X.n)
size(X::SparseBinMatrix, d::Int) = d==1 ? X.m : X.n

size(X::ParallelSBM) = (X.m, X.n)
size(X::ParallelSBM, d::Int) = d==1 ? X.m : X.n

import Base.A_mul_B!

## multiplication: y = A * x
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

function A_mul_B!{Tx}(y::SharedArray{Tx,1}, A::ParallelSBM, x::SharedArray{Tx,1})
  A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
  A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
  y[1:end] = zero(Tx)
  @sync begin
    for p in 1:length(A.pids)
      pid = A.pids[p]
      if pid != myid() || np == 1
        @async begin
          remotecall_wait(pid, partmul_ref, y, A.sbms[p], A.logic[p], x)
        end
      end
    end
  end
  ## done
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
  ## doing tmp = A * x
  @sync begin
    for p in 1:length(A.pids)
      pid = A.pids[p]
      if pid != myid() || np == 1
        @async begin
          remotecall_wait(pid, partmul_ref, tmp, A.sbms[p], A.logic[p], x)
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
          remotecall_wait(pid, partmul_t_ref, y, A.sbms[p], A.logic[p], tmp)
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
        if pid != myid() || np == 1
          @async begin
            ptime[p, i] += remotecall_fetch(pid, partmul_time, y, A.sbms[p], A.logic[p], x)
          end
        end
      end
    end
  end
  return ptime
end

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

function partmul_time{Tx}(y::SharedArray{Tx,1}, Aref::RemoteRef, logicref::RemoteRef, x::SharedArray{Tx,1})
  A     = fetch(Aref)::SparseBinMatrix
  logic = fetch(logicref)::ParallelLogic
  tic();
  partmul(y, A, logic, x)
  return toq()
end

function partmul_ref{Tx}(y::SharedArray{Tx,1}, Aref::RemoteRef, logicref::RemoteRef, x::SharedArray{Tx,1})
  A     = fetch(Aref)::SparseBinMatrix
  logic = fetch(logicref)::ParallelLogic
  partmul(y, A, logic, x)
end

function partmul_t_ref{Tx}(y::SharedArray{Tx,1}, Aref::RemoteRef, logicref::RemoteRef, x::SharedArray{Tx,1})
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
      #break
    end
  end
  return nothing
end

function At_mul_B!{Tx}(y::SharedArray{Tx,1}, A::ParallelSBM, x::SharedArray{Tx,1})
  # TODO
end

######## Hilbert ordering ########

## convert (x,y) to d
function xy2d (n, x, y)
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
#end

###################### SparseBinCSR #######################

type SparseBinMatrixCSR
  m::Int
  n::Int
  col_ind::Vector{Int32}
  row_ptr::Vector{Int32}
end

function SparseBinMatrixCSR(rows::Vector{Int32}, cols::Vector{Int32})
  rsorted = sortperm(rows)
  m = convert(Int, maximum(rows))
  n = convert(Int, maximum(cols))
  row_ptr = zeros(Int32, m+1)
  row_ptr[1:end] = length(rows)+1
  rows2   = rows[rsorted]
  prev = zero(Int32)
  for i = one(Int32):convert(Int32,length(rows))
    while rows2[i] > prev
      prev += one(Int32)
      row_ptr[prev] = i
    end
  end
  return SparseBinMatrixCSR(m, n, cols[rsorted], row_ptr)
end

size(A::SparseBinMatrixCSR) = (A.m, A.n)
size(X::SparseBinMatrixCSR, d::Int) = d==1 ? X.m : X.n

function A_mul_B!{Tx}(y::AbstractArray{Tx,1}, A::SparseBinMatrixCSR, x::AbstractArray{Tx,1})
    A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
    A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
    fill!(y, zero(Tx) )
    zro = zero(Tx)
    row_ptr = A.row_ptr
    col_ind = A.col_ind
    @inbounds for row = 1:A.m
        tmp = zro
        r1  = row_ptr[row]
        r2  = row_ptr[row+1] - 1
        for i = r1:r2
          tmp += x[col_ind[i]]
        end
        y[row] = tmp
    end
    return
end
