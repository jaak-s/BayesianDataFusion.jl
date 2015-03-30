export read_ecfp, read_sparse, read_rowcol, read_rowcol_binary, filter_rare, write_binary

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
            push!( rows, convert(Int32, parse(a[1])) )
            push!( cols, convert(Int32, parse(a[2])) )
        end
    end
    return rows, cols
end

function read_rowcol_binary(filename)
    rows = Int32[]
    cols = Int32[]
    open(filename) do f
        len = read(f, Int64)
        return read(f, Int32, (2, len))
    end
end

function read_sparse(filename)
    rows = Int32[]
    cols = Int32[]
    open(filename) do f
        for line in eachline(f)
            a = split(line, ",")
            push!( rows, convert(Int32, parse(a[1])) )
            push!( cols, convert(Int32, parse(a[2])) )
        end
    end
    return sparse(rows, cols, 1f0)
end

function filter_rare(X::SparseMatrixCSC, nmin)
    featn = vec(sum(X, 1))
    return X[:, featn .>= nmin]
end

function write_binary(filename, X1::Vector{Int32}, X2::Vector{Int32})
    length(X1) == length(X2) && error("X1 and X2 must have same length.")
    open(filename, "w") do f
        write(f, length(X1))
        for i = 1:length(X1)
            write(f, X1[i])
            write(f, X2[i])
        end
    end
end
