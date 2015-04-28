#module ParallelMatrix

export SparseBinMatrix, ParallelLogic, ParallelSBM, balanced_parallelsbm
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

  ## block mutex: every 1:8:(nblocks*8+1)
  mutex::SharedArray{Int,1}
  ## array to store error
  error::SharedArray{Int,1}
end

type ParallelSBM
  m::Int64
  n::Int64
  pids::Vector{Int64}
  sbms::Vector{RemoteRef}  ## SparseBinMatrix
  logic::Vector{RemoteRef} ## ParallelLogic

  error::SharedArray{Int,1} ## keeping sync errors
end

function ParallelSBM(rows::Vector{Int32}, cols::Vector{Int32}, pids::Vector{Int}; weights=ones(length(pids)), m=maximum(rows), n=maximum(cols), numblocks=length(pids)+2 )
  length(rows) == length(cols) || throw(DimensionMismatch("length(rows) must equal length(cols)"))

  ps = ParallelSBM(m, n, pids, RemoteRef[], RemoteRef[], SharedArray(Int,8))
  ranges  = make_lengths(length(rows), weights)
  mblocks = make_blocks(m, convert(Int32, numblocks) )
  nblocks = make_blocks(n, convert(Int32, numblocks) )
  mutex   = make_mutex(numblocks)
  merror  = SharedArray(Int, 8)
  for i in 1:length(pids)
    sbm = SparseBinMatrix(m, n, rows[ranges[i]], cols[ranges[i]])
    sbm_ref = @spawnat pids[i] fetch(sbm)
    push!(ps.sbms, sbm_ref)
    mb = find([! isempty(intersect(sbm.mrange, i)) for i in mblocks])
    nb = find([! isempty(intersect(sbm.nrange, i)) for i in nblocks])
    pl_ref = @spawnat pids[i] ParallelLogic(mblocks, nblocks, mb, nb, zeros(m), zeros(n), mutex, merror)
    push!(ps.logic, pl_ref)
  end
  return ps
end

function balanced_parallelsbm(rows::Vector{Int32}, cols::Vector{Int32}, pids::Vector{Int}; numblocks=length(pids)+2, niter=4)
  weights = ones(length(pids))
  y = SharedArray(Float64, convert(Int, maximum(rows)) )
  x = SharedArray(Float64, convert(Int, maximum(cols)) )
  local psbm
  for i = 1:niter
    psbm   = ParallelSBM(rows, cols, pids, numblocks=numblocks, weights=weights)
    ctimes = A_mul_B!_time(y, psbm, x, 5)
    weights = weights ./ ctimes
    weights = weights / sum(weights)
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
  if A.error[1] != 0
    error("Mutex error occured")
  end
  ## done
end

function A_mul_B!_time{Tx}(y::SharedArray{Tx,1}, A::ParallelSBM, x::SharedArray{Tx,1}, ntimes::Int)
  A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
  A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
  ptime = zeros(length(A.pids))
  for i = 1:ntimes+1
    y[1:end] = zero(Tx)
    ## clearing warmup results
    if i == 2
      ptime[1:end] = 0.0
    end
    @sync begin
      for p in 1:length(A.pids)
        pid = A.pids[p]
        if pid != myid() || np == 1
          @async begin
            ptime[p] += remotecall_fetch(pid, partmul_time, y, A.sbms[p], A.logic[p], x)
          end
        end
      end
    end
  end
  if A.error[1] != 0
    error("Mutex error occured")
  end
  return ptime / ntimes
end

make_mutex(nblocks) = SharedArray(Int, nblocks*8)

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
  addshared!(y, ylocal, logic.mutex, logic.mblocks, logic.mblock_order, logic.error)
  return nothing
end

## does y += x
function addshared!{Tx}(y::SharedArray{Tx,1}, x::AbstractArray{Tx,1}, mutex::SharedArray{Int,1}, ranges, order, mutex_error::SharedArray{Int,1})
  blocks = copy(order)
  nblocks = length(blocks)
  pid = myid()
  while true
    i = findfirst(blocks)
    i <= 0 && return ## done
    for j = i:nblocks
      blocks[j] == 0 && continue
      ## try to get a lock
      if ! ask_for_lock!(mutex, blocks[j], pid)
        continue
      end
      ## copying result to shared array
      add!(y, x, ranges[blocks[j]])
      if ! release_lock!(mutex, blocks[j], pid)
        ## release failed, some error
        mutex_error[1] = 1
        return
      end
      blocks[j] = 0
      break
    end
  end
  return nothing
end

## ask for a lock for block
function ask_for_lock!(mutex, block::Int, pid::Int)
  i = 1 + (block-1)*8
  m = mutex[i]
  if m != 0
    return m == pid
  end
  mutex[i] = pid
  busywait(mutex, i, pid, 30)
  return mutex[i] == pid
end

## releases lock, if not correct pid returns false
function release_lock!(mutex, block::Int, pid::Int)
  i = 1 + (block-1)*8
  mutex[i] != pid && return false
  mutex[i] = 0
  return true
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
