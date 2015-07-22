using DataFrames

include("IndexedDF.jl")
typealias SparseMatrix SparseMatrixCSC{Float64, Int64} 

export RelationData, addRelation!
export Relation, numData, numTest, assignToTest!
export Entity, toStr, normalizeFeatures!, normalizeRows!
export EntityModel
export load_mf1c
export setPrecision!

type EntityModel
  sample::Matrix{Float64}  ## latent vectors (each column is one instance)

  mu    ::Vector{Float64}  ## mean
  Lambda::Matrix{Float64}  ## Precision matrix
  beta  ::Matrix{Float64}  ## parameter linking features to latent

  mu0   ::Vector{Float64}  ## Hyper-prior mean for NormalWishart
  b0    ::Float64          ## Hyper-prior for NormalWishart
  WI    ::Matrix{Float64}  ## Hyper-prior for NormalWishart (inverse of W)

  EntityModel() = new()
  EntityModel(num_latent::Int, num_instances::Int) = new(
    zeros(num_latent, num_instances), ## sample
    zeros(num_latent),    ## mu
    eye(num_latent),      ## Lambda
    zeros(0, num_latent), ## beta
    zeros(num_latent),    ## mu0
    2.0,                  ## b0
    eye(num_latent)       ## WI
  )
end

type Entity{FT,R}
  F::FT
  FF
  Frefs::Vector{RemoteRef}
  relations::Vector{R}
  count::Int64
  name::String

  lambda_beta::Float64
  lambda_beta_sample::Bool
  mu::Float64   ## Hyper-prior for lambda_beta
  nu::Float64   ## Hyper-prior for lambda_beta

  model::EntityModel
  Entity(F, relations::Vector{R}, count::Int64, name::String, lb::Float64=1.0, lb_sample::Bool=true, mu=1.0, nu=1.0) = new(F, zeros(0,0), RemoteRef[], relations, count, name, lb, lb_sample, mu, nu)
end

Entity(name::String; F=zeros(0,0), lambda_beta=1.0) = Entity{Any,Relation}(F::Any, Relation[], 0, name, lambda_beta)

## initializes the model parameters
function initModel!(entity::Entity, num_latent::Int64; lambda_beta::Float64 = NaN)
  m = EntityModel()
  entity.model = m

  m.sample = zeros(num_latent, entity.count)
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

  if ! isnan(lambda_beta)
    entity.lambda_beta = lambda_beta
  end

  return nothing
end

hasFeatures(entity::Entity) = ! isempty(entity.F)

function toStr(en::Entity)
  if ! isdefined(en, :model)
    return string(en.name[1:min(3,end)], "[]")
  end
  return string(
    en.name[1:min(3,end)],
    "[",
       @sprintf("U:%6.2f", vecnorm(en.model.sample)),
       hasFeatures(en) ? @sprintf(" β:%3.2f", vecnorm(en.model.beta)) :"",
       hasFeatures(en) && en.lambda_beta_sample ? @sprintf(" λ=%1.1f", en.lambda_beta) :"",
    "]")
end

type RelationModel
  alpha_sample::Bool
  alpha_nu0::Float64
  alpha_lambda0::Float64

  lambda_beta::Float64

  alpha::Float64
  beta::Vector{Float64}
  mean_value::Float64
end

RelationModel(alpha::Float64, lambda_beta::Float64=1.0) = RelationModel(false, 0.0, 1.0, lambda_beta, alpha, zeros(0), 0.0)
RelationModel() = RelationModel(true, 2, 1.0, 1.0, NaN, zeros(0), 0.0)

type RelationTemp
  linear_values::Vector{Float64} ## mean_value + F * beta
  FF::Matrix{Float64}

  RelationTemp() = new()
end

type Relation
  data::IndexedDF
  F
  entities::Vector{Entity}
  name::String

  test_vec::DataFrame
  test_F
  test_label::Vector{Bool}
  class_cut::Float64

  model::RelationModel
  temp::RelationTemp

  Relation(data::IndexedDF, name::String, class_cut, alpha) = new(data, (), Entity[], name, data.df[Int[],:], (), Bool[], class_cut, RelationModel(alpha))
  Relation(data::IndexedDF, name::String, class_cut=0.0) = new(data, (), Entity[], name, data.df[Int[],:], (), Bool[], class_cut, RelationModel())
  ## Relation with already setup entities
  function Relation(data::DataFrame, name::String, entities=Entity[]; class_cut=0.0,
                    dims = Int64[maximum(data[:,i]) for i in 1 : size(data,2)-1])
    if ! isempty(entities)
      size(data, 2) - 1 == length(entities) || throw(ArgumentError("data has $(size(data,2)) columns but needs to have $(length(entities)) which is number of entities + 1"))
      for i in 1:length(entities)
        en = entities[i]
        ## update the counts of the entity
        if en.count == 0
          en.count = dims[i]
        elseif en.count > dims[i]
          dims[i] = en.count
        elseif en.count < dims[i]
          throw(ArgumentError("Entity $(en.name) has smaller count $(en.count) than the largest id in the data $(dims[i]). Set entity.count manually before creating the relation."))
        end
      end
    end
    return new(IndexedDF(data, dims), (), entities, name, data[Int[],:], (), Bool[], class_cut, RelationModel())
  end
end

## creating Relation from sparse matrix
function Relation(data::SparseMatrixCSC, name::String, entities=Entity[]; class_cut=0.0)
  length(entities) == 2 || throw(ArgumentError("For matrix relation the number of entities has to be 2."))
  U, V, X = findnz(data)
  dims = Int[size(data,1), size(data,2)]
  df = DataFrame(E1=U, E2=V, values=X)
  return Relation(df, name, entities, class_cut = class_cut, dims = dims)
end

import Base.size
size(r::Relation) = tuple([length(x) for x in r.data.index]...)
size(r::Relation, d::Int) = length(r.data.index[d])
numData(r::Relation) = nnz(r.data)
numTest(r::Relation) = size(r.test_vec, 1)
hasFeatures(r::Relation) = ! isempty(r.F)
function toStr(r::Relation)
  if ! isdefined(r, :model)
    return string(r.name[1:min(4,end)], "[]")
  end
  return string(
    r.name[1:min(4,end)],
    "[",
       @sprintf("α=%2.1f", r.model.alpha),
       hasFeatures(r) ? @sprintf(" β:%2.1f", vecnorm(r.model.beta)) :"",
    "]")
end

function assignToTest!(r::Relation, ntest::Int64)
  test_id  = sample(1:size(r.data.df,1), ntest; replace=false)
  assignToTest!(r, test_id)
end

function setPrecision!(r::Relation, precision)
  r.model.alpha = precision
end

function assignToTest!(r::Relation, test_id::Vector{Int64})
  test_vec = r.data.df[test_id, :]
  r.data   = removeSamples(r.data, test_id)
  r.test_vec    = test_vec
  r.test_label  = r.test_vec[:,end] .< r.class_cut
  if hasFeatures(r)
    r.test_F = r.F[test_id,:]
    train    = ones(Bool, size(r.F, 1) )
    train[test_id] = false
    r.F      = r.F[train,:]
  end
  nothing
end

type RelationData
  entities::Vector{Entity}
  relations::Vector{Relation}

  RelationData() = new( Entity[], Relation[] )

  function RelationData(Am::IndexedDF; feat1=zeros(0,0), feat2=zeros(0,0), entity1="E1", entity2="E2", relation="Rel", ntest=0, class_cut=log10(200), alpha=5.0, alpha_sample=false, lambda_beta=1.0)
    r  = alpha_sample ?Relation(Am, relation, class_cut) :Relation(Am, relation, class_cut, alpha)
    e1 = Entity{isempty(feat1) ? Any :typeof(feat1), Relation}( feat1, [r], size(r,1), entity1, lambda_beta )
    e2 = Entity{isempty(feat2) ? Any :typeof(feat2), Relation}( feat2, [r], size(r,2), entity2, lambda_beta )
    if ! isempty(feat1) && size(feat1,1) != size(r,1)
      throw(ArgumentError("Number of rows in feat1 $(size(feat1,1)) must equal number of rows in the relation $(size(Am,1))"))
    end
    if ! isempty(feat2) && size(feat2,1) != size(Am,2)
      throw(ArgumentError("Number of rows in feat2 $(size(feat2,1)) must equal number of columns in the relation $(size(Am,2))"))
    end
    push!(r.entities, e1)
    push!(r.entities, e2)
    return new( Entity[e1, e2], Relation[r] )
  end
end

function RelationData(Am::DataFrame; rname="R1", class_cut=log10(200), alpha=5.0)
  dims = Int64[maximum(Am[:,i]) for i in 1 : size(Am,2)-1]
  idf  = IndexedDF(Am, dims)
  rd   = RelationData()
  push!(rd.relations, Relation(idf, rname, class_cut, alpha))
  for d in 1:length(dims)
    en = Entity{Any, Relation}( zeros(0,0), [rd.relations[1]], size(idf,d), string(names(Am)[d]))
    push!(rd.entities, en)
    push!(rd.relations[1].entities, en)
  end
  return rd
end

function RelationData(M::SparseMatrixCSC{Float64,Int64}; kw...)
  dims = size(M)
  cols = rep(1:size(M,2), M.colptr[2:end] - M.colptr[1:end-1])
  df   = DataFrame( row=M.rowval, col=cols, value=nonzeros(M) )
  idf  = IndexedDF(df, dims)
  return RelationData(idf; kw...)
end

function RelationData(r::Relation)
  rd = RelationData()
  addRelation!(rd, r)
  return rd
end

## computes F * beta
function F_mul_beta{F}(en::Entity{F,Relation})
  if ! isempty(en.Frefs)
    return Frefs_mul_B(en.Frefs, en.model.beta)
  else
    return en.F * en.model.beta
  end
end

## computes F' * beta
function Ft_mul_B{F}(en::Entity{F,Relation}, B::Matrix{Float64})
  if ! isempty(en.Frefs)
    return Frefs_t_mul_B(en.Frefs, B)
  else
    return At_mul_B(en.F, B)
  end
end

function reset!(data::RelationData, num_latent; lambda_beta=NaN, compute_ff_size = 6500, cg_pids=Int[myid()])
  for en in data.entities
    initModel!(en, num_latent, lambda_beta = lambda_beta)
    if hasFeatures(en)
      if size(en.F,2) <= compute_ff_size
        en.FF = full(At_mul_B(en.F, en.F))
      else
        ## setup CG on julia threads
        init_Frefs!(en, num_latent, cg_pids)
      end
    end
  end
  for r in data.relations
    r.model.mean_value    = valueMean(r.data)
    r.temp = RelationTemp()
    if hasFeatures(r) && size(r.F,2) <= compute_ff_size
      r.temp.linear_values = r.model.mean_value * ones(numData(r))
      r.temp.FF = full(r.F' * r.F)
    end
  end
end

function init_Frefs!(en::Entity, num_latent::Int, pids::Vector{Int})
  Fns = nonshared(en.F)
  empty!(en.Frefs)
  for i = 1:min(length(pids), num_latent)
    pid = pids[i]
    push!(en.Frefs, @spawnat pid copyto(Fns, Int[pid]))
  end
end

function init_Frefs!(en::Entity, num_latent::Int, pids::Vector{Vector{Int}})
  Fns = nonshared(en.F)
  empty!(en.Frefs)
  for i = 1:min(length(pids), num_latent)
    pid   = pids[i][1]
    mulpids = pids[i][2:end]
    isempty(mulpids) && push!(mulpids, pid)
    push!(en.Frefs, @spawnat pid copyto(Fns, mulpids))
  end
end

function init_Frefs!(en::Entity, num_latent::Int, pids::Dict{Int, Vector{Int}})
  Fns = nonshared(en.F)
  empty!(en.Frefs)
  for e in pids
    pid     = e[1]
    mulpids = e[2]
    push!(en.Frefs, @spawnat pid copyto(Fns, mulpids))
  end
end

function addRelation!(rd::RelationData, r::Relation)
  if length(size(r)) != length(r.entities)
    throw(ArgumentError("Relation has $(length(r.entities)) entities but its data implies $(size(r))."))
  end
  push!(rd.relations, r)
  ## adding entities
  for i in 1:length(r.entities)
    en = r.entities[i]
    if en.count == 0
      ## updating entity count
      en.count = size(r, i)
    elseif en.count != size(r, i)
      throw(ArgumentError("Entity $(en.name) has $(en.count) instances, relation $(r.name) has data for $(size(r.entities, i))."))
    end
    if ! any(rd.entities .== en)
      push!(rd.entities, en)
    end
    if ! any(en.relations .!= r)
      push!(en.relations, r)
    end
  end
  return nothing
end

import Base.show
function show(io::IO, rd::RelationData)
  println(io, "[Relations]")
  for r in rd.relations
    @printf(io, "%10s: %s, #known = %d, #test = %d, α = %s", r.name, join([e.name for e in r.entities], "--"), numData(r), numTest(r), r.model.alpha_sample ?"sample" :@sprintf("%.2f", r.model.alpha))
    hasFeatures(r) && @printf(io, ", #feat = %d", size(r.F,2))
    @printf(io, "\n")
  end

  println(io, "[Entities]")
  for en in rd.entities
    @printf(io, "%10s: %6d ", en.name, en.count)
    if hasFeatures(en)
      print(io, "with ", size(en.F,2), " features (λ = ",
                en.lambda_beta_sample ? "sample" :@sprintf("%1.1f", en.lambda_beta),")")
    else
      print(io, "with no features")
    end
    println(io)
  end
end

function normalizeFeatures!(entity::Entity)
  diagsq  = sqrt(vec( sum(entity.F .^ 2,1) ))
  entity.F  = entity.F * spdiagm(1.0 ./ diagsq)
  return
end

function normalizeRows!(entity::Entity)
  diagf    = sqrt(vec( sum(entity.F.^2, 2) ))
  entity.F = spdiagm( 1.0 ./ diagf ) * entity.F
end

function load_mf1c(;ic50_file     = "chembl_19_mf1c/chembl-IC50-346targets.csv",
                   cmp_feat_file  = "chembl_19_mf1c/chembl-IC50-compound-feat.csv",
                   normalize_feat = false,
                   alpha_sample   = false)
  ## reading IC50 matrix
  X = readtable(ic50_file, header=true)
  rename!(X, [:row, :col], [:compound, :target])

  dims = [maximum(X[:compound]), maximum(X[:target])]

  X[:, :value] = log10(X[:, :value]) + 1e-5
  idx          = sample(1:size(X,1), int(floor(20/100 * size(X,1))); replace=false)
  probe_vec    = X[idx,:]
  X            = X[setdiff(1:size(X,1), idx), :]
  
  ## reading feature matrix
  feat = readtable(cmp_feat_file, header=true)
  F    = sparse(feat[:compound], feat[:feature], 1.0)

  #Am = sparse( X[:compound], X[:target], X[:value])
  Xi = IndexedDF(X, dims)
  
  ## creating data object
  data = RelationData(Xi, feat1 = F, alpha_sample = alpha_sample)
  data.relations[1].test_vec    = probe_vec
  data.relations[1].test_label  = data.relations[1].test_vec[:,3] .< log10(200)

  if normalize_feat != false
    if normalize_feat == "rows"
      normalizeRows!(data.entities[1])
    else
      normalizeFeatures!(data.entities[1])
    end
  end
  
  return data
end
