using DataFrames

include("IndexedDF.jl")
typealias SparseMatrix SparseMatrixCSC{Float64, Int64} 

type EntityModel
  sample::Matrix{Float64}  ## latent vectors (each row is one instance)

  mu    ::Vector{Float64}  ## mean
  Lambda::Matrix{Float64}  ## Precision matrix
  beta  ::Matrix{Float64}  ## parameter linking features to latent

  mu0   ::Vector{Float64}  ## Prior mean for NormalWishart
  b0    ::Float64          ## Prior for NormalWishart
  WI    ::Matrix{Float64}  ## Prior for NormalWishart
  lambda_beta::Float64     ## Prior for beta

  EntityModel() = new()
end

type Entity{FT,R}
  F::FT
  relations::Vector{R}
  count::Int64
  name::String
  model::EntityModel
  Entity{FT,R}(F::FT, relations::Vector{R}, count::Int64, name::String) = new(F, relations, count, name)
end

## initializes the model parameters
function initModel!(entity::Entity, num_latent::Int64; lambda_beta::Float64 = 1.0)
  m = EntityModel()
  m.sample = zeros(entity.count, num_latent)
  m.mu     = zeros(num_latent)
  m.Lambda = eye(num_latent)
  if hasFeatures(entity)
    m.beta = zeros( size(entity.F, 2), num_latent )
  else
    m.beta = zeros( 0, num_latent )
  end

  m.mu0    = zeros(num_latent)
  m.b0     = 2.0
  m.WI     = eye(num_latent)
  m.lambda_beta = lambda_beta

  entity.model = m
  return nothing
end

hasFeatures(entity::Entity) = ! isempty(entity.F)

type Relation
  data::IndexedDF
  entities::Vector{Entity}
  name::String

  test_vec::DataFrame
  test_label::Vector{Bool}
  mean_rating::Float64

  class_cut::Float64

  Relation(data::IndexedDF, name::String, class_cut=0.0) = new(data, Entity[], name, data.df[[],:], Bool[], valueMean(data), class_cut)
end

import Base.size
size(r::Relation) = [length(x) for x in r.data.index]
size(r::Relation, d::Int) = length(r.data.index[d])
numData(r::Relation) = nnz(r.data)
numTest(r::Relation) = size(r.test_vec, 1)

function assignToTest!(r::Relation, ntest::Int64)
  test_id  = sample(1:size(r.data.df,1), ntest; replace=false)
  test_vec = array(r.data.df[test_id, :])
  r.data   = removeSamples(r.data, test_id)
  r.test_vec    = test_vec
  r.test_label  = r.test_vec[:,end] .< r.class_cut
  r.mean_rating = valueMean(r.data)
  nothing
end

type RelationData
  entities::Vector{Entity}
  relations::Vector{Relation}

  function RelationData(Am::IndexedDF; feat1=(), feat2=(), entity1="compound", entity2="protein", relation="IC50", ntest=0, class_cut=log10(200))
    r  = Relation( Am, relation, class_cut )
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

function RelationData(Am::DataFrame; kw...)
  dims = [maximum(Am[:,i]) for i in 1 : size(Am,2)-1]
  idf  = IndexedDF(Am, dims)
  return RelationData(idf; kw...)
end

function RelationData(M::SparseMatrixCSC{Float64,Int64}; kw...)
  dims = size(M)
  cols = rep([1:size(M,2)], M.colptr[2:end] - M.colptr[1:end-1])
  df   = DataFrame( row=M.rowval, col=cols, value=nonzeros(M) )
  idf  = IndexedDF(df, dims)
  return RelationData(idf; kw...)
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

  if normalize_feat
    normalizeFeatures!(data.entities[1])
  end
  
  return data
end