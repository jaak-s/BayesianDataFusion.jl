export SparseBinMatrixCSR, ParallelBinCSR
using Compat

###################### SparseBinMatrixCSR #######################

type SparseBinMatrixCSR
  m::Int
  n::Int
  col_ind::Vector{Int32}
  row_ptr::Vector{Int32}
end

@compat type ParallelBinCSR
  m::Int
  n::Int
  pids::Vector{Int64}
  csrs::Vector{Future}
  mranges::Vector{StepRange{Int64,Int64}}
  blocksize::Int64
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

import Base.size
size(A::SparseBinMatrixCSR) = (A.m, A.n)
size(X::SparseBinMatrixCSR, d::Int) = d==1 ? X.m : X.n

import Base.show
function show(io::IO, csr::SparseBinMatrixCSR)
  println(io, "$(csr.m) x $(csr.m) binary sparse matrix (CSR) with $(length(csr.col_ind)) entries.")
end

import Base.A_mul_B!
function A_mul_B!{Tx}(y::AbstractArray{Tx,1}, A::SparseBinMatrixCSR, x::AbstractArray{Tx,1})
    A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
    A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
    zro = zero(Tx)
    row_ptr = A.row_ptr
    col_ind = A.col_ind
    @inbounds for row = 1:A.m
        tmp = zro
        for i = row_ptr[row]:row_ptr[row+1]-1
          tmp += x[col_ind[i]]
        end
        y[row] = tmp
    end
    return
end

######## ParallelBinCSR #########
function ParallelBinCSR(rows::Vector{Int32}, cols::Vector{Int32}, pids::Vector{Int}, blocksize=64)
  csr = SparseBinMatrixCSR(rows, cols)
  npids  = length(pids)
  ranges = StepRange{Int,Int}[ 1+(i-1)*blocksize:npids*blocksize:size(csr,1) for i in 1:npids ]
  @compat pcsr = ParallelBinCSR(csr.m, csr.n, pids, Future[], ranges, blocksize)
  for i in 1:npids
    ref = @spawnat pids[i] fetch(csr)
    push!(pcsr.csrs, ref)
  end
  return pcsr
end

function A_mul_B!{Tx}(y::SharedArray{Tx,1}, A::ParallelBinCSR, x::SharedArray{Tx,1})
  A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
  A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))

  @sync begin
    for p in 1:length(A.pids)
      pid = A.pids[p]
      if pid != myid() || np == 1
        @async begin
          remotecall_wait(A_mul_B_part_ref, pid, y, A.csrs[p], x, A.mranges[p], A.blocksize)
        end
      end
    end
  end
  return nothing
end

@compat function A_mul_B_part_ref{Tx}(y::SharedArray{Tx,1}, Aref::Future, x::SharedArray{Tx,1}, range::StepRange{Int,Int}, blocksize::Int)
  A = fetch(Aref)::SparseBinMatrixCSR
  A_mul_B_range!(y, A, x, range, blocksize)
  return nothing
end

function A_mul_B_range!{Tx}(y::AbstractArray{Tx,1}, A::SparseBinMatrixCSR, x::AbstractArray{Tx,1}, range, blocksize::Int)
    A.n == length(x) || throw(DimensionMismatch("A.n=$(A.n) must equal length(x)=$(length(x))"))
    A.m == length(y) || throw(DimensionMismatch("A.m=$(A.m) must equal length(y)=$(length(y))"))
    zro = zero(Tx)
    row_ptr = A.row_ptr
    col_ind = A.col_ind
    @inbounds for b0 = range, row=b0:min(b0+blocksize-1, A.m)
        tmp = zro
        for i = row_ptr[row]:row_ptr[row+1]-1
          tmp += x[col_ind[i]]
        end
        y[row] = tmp
    end
    return
end
