import Base.At_mul_B!

if v"0.4.0" >= v"0.4.0-dev+4100"

function At_mul_B!{Tv,Ti}(y::AbstractMatrix{Tv}, A::SparseMatrixCSC{Tv,Ti}, x::AbstractMatrix{Tv})
    A.n == size(y, 1) || throw(DimensionMismatch("A.n==$(A.n) must equal size(y,1)==$(size(y,1))"))
    A.m == size(x, 1) || throw(DimensionMismatch("A.m==$(A.m) must equal size(x,1)==$(size(x,1))"))
    size(y,2) == size(x,2) || throw(DimensionMismatch("size(y,2)==$(size(y,2)) must equal size(x,2)==$(size(x,2))"))
    nzv = A.nzval
    rv  = A.rowval
    colptr = A.colptr
    zro = zero(Tv)
    @inbounds begin
        for k = 1:size(y,2)
            for i = 1 : A.n
                tmp = zro
                for j = colptr[i] : (colptr[i+1]-1)
                    tmp += nzv[j]*x[rv[j],k]
                end
                y[i,k] = tmp
            end
        end
    end
    y
end

end
