using DataFrames

export IndexedDF, getData, getCount, removeSamples, getValues, valueMean

type IndexedDF
  df::DataFrame
  index::Vector{Vector{Vector{Int64}}}

  function IndexedDF(df::DataFrame, dims::Vector{Int64})
    ## indexing all columns D - 1 columns (integers)
    index = [ [Int64[] for j in 1:i] for i in dims ]
    for i in 1:size(df, 1)
      for mode in 1:length(dims)
        j = df[i, mode]
        push!(index[mode][j], i)
      end
    end
    new(df, index)
  end
end

IndexedDF(df::DataFrame, dims::Tuple) = IndexedDF(df, Int64[i for i in dims])
IndexedDF(df::DataFrame) = IndexedDF(df, Int64[maximum(df[:,i]) for i in 1 : size(df,2)-1])

valueMean(idf::IndexedDF) = mean(idf.df[:,end])
import Base.size
size(idf::IndexedDF) = tuple( [length(i) for i in idf.index]... )
size(idf::IndexedDF, i::Int64) = length(idf.index[i])

import Base.nnz
nnz(idf::IndexedDF) = size(idf.df, 1)

function removeSamples(idf::IndexedDF, samples)
  df = idf.df[ setdiff(1:size(idf.df, 1), samples), :]
  return IndexedDF(df, size(idf))
end

getValues(idf::IndexedDF) = vec(array(idf.df[:, end]))
getMode(idf::IndexedDF, mode::Int64) = idf.df[:, mode]
getData(idf::IndexedDF, mode::Int64, i::Int64)  = idf.df[ idf.index[mode][i], :]
getCount(idf::IndexedDF, mode::Int64, i::Int64) = length( idf.index[mode][i] )
getI(idf::IndexedDF, mode::Int64, i::Int64)     = idf.index[mode][i]
