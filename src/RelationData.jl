using DataFrames

typealias SparseMatrix SparseMatrixCSC{Float64, Int64} 

type Entity{FT,R}
  F::FT
  relations::Vector{R}
  name::String
end

hasFeatures(entity::Entity) = ! isempty(entity.F)

type Relation
  data::Vector{SparseMatrix} ## sparse matrix for each mode
  entities::Vector{Entity}
  name::String

  test_vec
  test_ratings
  mean_rating::Float64

  Relation(data::SparseMatrix, name::String) = new({data,data'}, Entity[], name)
end

import Base.size
size(r::Relation) = size(r.data[1])

type RelationData
  entities::Vector{Entity}
  relations::Vector{Relation}
  function RelationData(Am::SparseMatrix; feat1=(), feat2=(), entity1="compound", entity2="protein", relation="IC50")
    r  = Relation( Am, relation )
    e1 = Entity{typeof(feat1),Relation}( feat1, [r], entity1 )
    e2 = Entity{typeof(feat2),Relation}( feat2, [r], entity2 )
    push!(r.entities, e1)
    push!(r.entities, e2)
    return new( {e1, e2}, {r} )
  end
end

type RelationDataX{XT,YT}
  F::XT
  Ft::XT
  Am::YT
  Au::YT
  num_p::Int
  num_m::Int
  probe_vec
  ratings_test
  mean_rating::Float64
  RelationDataX(F, Am) = new(F, F', Am, Am', size(Am,1), size(Am,2))
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

  X[:, :value] = log10(X[:, :value]) + 1e-5
  idx          = sample(1:size(X,1), int(floor(20/100 * size(X,1))); replace=false)
  probe_vec    = array(X[idx,:])
  X            = X[setdiff(1:size(X,1), idx), :]
  
  ## reading feature matrix
  feat = readtable(cmp_feat_file, header=true)
  F    = sparse(feat[:compound], feat[:feature], 1.0)

  Am = sparse( X[:compound], X[:target], X[:value])
  num_p, num_m = size(Am)
  
  ## creating data object
  data = RelationData(Am, feat1 = F)
  data.relations[1].test_vec     = probe_vec
  data.relations[1].test_ratings = data.relations[1].test_vec[:,3] .< log10(200)
  data.relations[1].mean_rating  = sum(Am) / size(X,1)

  if normalize_feat
    normalizeFeatures!(data.entities[1])
  end
  
  return data
end
