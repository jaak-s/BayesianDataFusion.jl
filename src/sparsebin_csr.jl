export SparseBinMatrixCSR

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

import Base.size
size(A::SparseBinMatrixCSR) = (A.m, A.n)
size(X::SparseBinMatrixCSR, d::Int) = d==1 ? X.m : X.n

import Base.A_mul_B!
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
