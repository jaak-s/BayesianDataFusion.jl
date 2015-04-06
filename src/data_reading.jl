export read_ecfp, read_sparse, read_rowcol
export read_binary_int32, filter_rare, write_binary_int32
export read_binary_float32
export read_sparse_float32

function read_ecfp(filename)
    i = 0
    next_fp = 1
    fp = Dict{Int32, Int32}()
    rows = Int32[]
    cols = Int32[]
    open(filename) do f
        for line in eachline(f)
            i += 1
            a = split(line, ",")
            for j = 2:length(a)
                fp_raw = parse(Int, a[j])
                local fp_id::Int32
                ## fetch fingerprint id range [1,...,max_fp]
                if haskey(fp, fp_raw)
                    fp_id = fp[fp_raw]
                else
                    fp_id      = next_fp
                    fp[fp_raw] = fp_id
                    next_fp += 1
                end
                push!(rows, i)
                push!(cols, fp_id)
            end
        end
    end
    println("Number of lines: $i")
    return rows, cols, fp
end

function read_rowcol(filename)
    rows = Int32[]
    cols = Int32[]
    open(filename) do f
        for line in eachline(f)
            a = split(line, ",")
            push!( rows, parse(Int32, a[1]) )
            push!( cols, parse(Int32, a[2]) )
        end
    end
    return rows, cols
end

function read_binary_int32(filename)
    open(filename) do f
        nrows = read(f, Int64)
        ncols = read(f, Int64)
        return read(f, Int32, (nrows, ncols))
    end
end

function read_binary_float32(filename)
    open(filename) do f
        nrows = read(f, Int64)
        ncols = read(f, Int64)
        return read(f, Float32, (nrows, ncols))
    end
end

function read_sparse_float32(filename)
  open(filename) do f
    nnz = read(f, Int64)
    rows = read(f, Int32, nnz)
    cols = read(f, Int32, nnz)
    vals = read(f, Float32, nnz)
    return rows, cols, vals
  end
end

function read_sparse(filename)
    rc = read_rowcol(filename)
    return sparse(rc[1], rc[2], 1f0)
end

function filter_rare(X::SparseMatrixCSC, nmin)
    featn = vec(sum(X, 1))
    return X[:, featn .>= nmin]
end

function write_binary_int32(filename, X::Matrix{Int32})
    write_binary_matrix(filename, X)
end

function write_binary_matrix(filename, X)
  open(filename, "w") do f
    write(f, size(X, 1))
    write(f, size(X, 2))
    write(f, X)
  end
end
