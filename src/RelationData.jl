using DataFrames

include("IndexedDF.jl")
typealias SparseMatrix SparseMatrixCSC{Float64, Int64} 

type Entity{FT,R}
  F::FT
  relations::Vector{R}
  count::Int
  name::String
end

hasFeatures(entity::Entity) = ! isempty(entity.F)

type Relation
  data::IndexedDF
  entities::Vector{Entity}
  name::String

  test_vec
  test_label::Vector{Bool}
  mean_rating::Float64

  Relation(data::IndexedDF, name::String) = new(data, Entity[], name)
end

import Base.size
size(r::Relation) = [length(x) for x in r.data.index]
size(r::Relation, d::Int) = length(r.data.index[d])
numData(r::Relation) = r.data.nnz
numTest(r::Relation) = size(r.test_vec, 1)

type RelationData
  entities::Vector{Entity}
  relations::Vector{Relation}
  function RelationData(Am::IndexedDF; feat1=(), feat2=(), entity1="compound", entity2="protein", relation="IC50")
    r  = Relation( Am, relation )
    e1 = Entity{typeof(feat1), Relation}( feat1, [r], size(r,1), entity1 )
    e2 = Entity{typeof(feat2), Relation}( feat2, [r], size(r,2), entity2 )
    if ! isempty(feat1) && size(feat1,1) != size(r,1)
      throw(ArgumentError("Number of rows in feat1 $(size(feat1,1)) must equal number of rows in the relation $(size(Am,1))"))
    end
    if ! isempty(feat2) && size(feat2,1) != size(Am,2)
      throw(ArgumentError("Number of rows in feat2 $(size(feat2,1)) must equal number of columns in the relation $(size(Am,2))"))
    end
    push!(r.entities, e1)
    push!(r.entities, e2)
    return new( {e1, e2}, {r} )
  end
end

import Base.show
function show(io::IO, rd::RelationData)
  println(io, "[Relations]")
  for r in rd.relations
    @printf(io, "%10s: %s, #known = %d, #test = %d\n", r.name, join([e.name for e in r.entities], "--"), numData(r), numTest(r))
  end
  println(io, "[Entities]")
  for en in rd.entities
    @printf(io, "%10s: %6d with %s features\n", en.name, en.count, hasFeatures(en) ?"$(size(en.F,2))" :"no")
  end
end

function normalizeFeatures!(entity::Entity)
  diagsq  = sqrt(vec( sum(entity.F .^ 2,1) ))
  entity.F  = entity.F * spdiagm(1.0 ./ diagsq)
  return
end

function load_mf1c(;ic50_file     = "chembl_19_mf1c/chembl-IC50-346targets.csv",
                   cmp_feat_file  = "chembl_19_mf1c/chembl-IC50-compound-feat.csv",
                   normalize_feat = false)
  ## reading IC50 matrix
  X = readtable(ic50_file, header=true)
  rename!(X, [:row, :col], [:compound, :target])

  dims = [maximum(X[:compound]), maximum(X[:target])]

  X[:, :value] = log10(X[:, :value]) + 1e-5
  idx          = sample(1:size(X,1), int(floor(20/100 * size(X,1))); replace=false)
  probe_vec    = array(X[idx,:])
  X            = X[setdiff(1:size(X,1), idx), :]
  
  ## reading feature matrix
  feat = readtable(cmp_feat_file, header=true)
  F    = sparse(feat[:compound], feat[:feature], 1.0)

  #Am = sparse( X[:compound], X[:target], X[:value])
  Xi = IndexedDF(X, dims)
  
  ## creating data object
  data = RelationData(Xi, feat1 = F)
  data.relations[1].test_vec    = probe_vec
  data.relations[1].test_label  = data.relations[1].test_vec[:,3] .< log10(200)
  data.relations[1].mean_rating = sum(X[:value]) / size(X,1)

  if normalize_feat
    normalizeFeatures!(data.entities[1])
  end
  
  return data
end
