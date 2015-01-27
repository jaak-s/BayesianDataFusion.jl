using DataFrames

type IndexedDF
  df::DataFrame
  index::Vector{Vector{Vector{Int64}}}
  nnz::Int64

  function IndexedDF(df::DataFrame, dims::Vector{Int64})
    ## indexing all columns D - 1 columns (integers)
    index = [ [Int64[] for j in 1:i] for i in dims ]
    for i in 1:size(df, 1)
      for mode in 1:length(dims)
        j = df[i, mode]
        push!(index[mode][j], i)
      end
    end
    new(df, index, size(df,1))
  end
end

IndexedDF(df::DataFrame, dims::Tuple) = IndexedDF(df, Int64[i for i in dims])

valueMean(idf::IndexedDF) = mean(idf.df[:,end])

function getData(idf::IndexedDF, mode::Int64, i::Int64)
  idf.df[ idf.index[mode][i], :]
end
