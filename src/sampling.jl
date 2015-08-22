using Iterators
using Distributions

export pred, pred_all
export solve_full

function pred(r::Relation, probe_vec::DataFrame, F)
  if ! hasFeatures(r)
    return udot(r, probe_vec) + r.model.mean_value
  end
  return udot(r, probe_vec) + F * r.model.beta + r.model.mean_value
end

function pred(r::Relation)
  udot(r, r.data.df) + (hasFeatures(r) ? r.temp.linear_values : r.model.mean_value)
end

## computes predictions sum(u1 .* u2 .* ... .* uR, 1) for relation r
function udot(r::Relation, probe_vec::DataFrame)
  U = r.entities[1].model.sample[ :, probe_vec[:,1] ]
  for i in 2:length(r.entities)
    U .*= r.entities[i].model.sample[ :, probe_vec[:,i] ]
  end
  return vec(sum(U, 1))
end

## faster udot (about 2x) for matrices 
function udot(r::Relation, probe_vec::Matrix{Integer})
  if length(size(r)) == 2
    ## special code for matrix
    num_latent = size(r.entities[1].model.sample, 1)
    result = zeros(size(probe_vec, 1))
    s1 = r.entities[1].model.sample
    s2 = r.entities[2].model.sample
    for i = 1:length(result)
      i1 = probe_vec[i,1]
      i2 = probe_vec[i,2]
      for k = 1:num_latent
        result[i] += s1[k,i1] * s2[k,i2]
      end
    end
    return result
  end
  U = r.entities[1].model.sample[:,probe_vec[:,1]]
  for i in 2:length(r.entities)
    U .*= r.entities[i].model.sample[:,probe_vec[:,i]]
  end
  return vec(sum(U, 1))
end

## computes predictions sum(u1 .* u2 .* ... .* uR, 2) for all points in relation r
function udot_all(r::Relation)
  ## matrix version:
  if length(r.entities) == 2
    return r.entities[1].model.sample' * r.entities[2].model.sample
  end

  ## TODO: make tensor version faster:
  U = zeros( Int64[en.count for en in r.entities]... )
  num_latent = length(r.entities[1].model.mu)
  for p in product( map(en -> 1:en.count, r.entities)... )
    x = ones(num_latent)
    for i in 1:length(p)
      x .*= r.entities[i].model.sample[:, p[i]]
    end
    U[p...] = sum(x)
  end
  return U
end

## predict all
function pred_all(r::Relation)
  if hasFeatures(r)
    error("Prediction of all elements is not possible when Relation has features.")
  end
  udot_all(r) + r.model.mean_value
end

function makeClamped(x, clamp::Vector{Float64})
  x2 = copy(x)
  x2[x2 .< clamp[1]] = clamp[1]
  x2[x2 .> clamp[2]] = clamp[2]
  return x2
end

function clamp!(x, clamp::Vector{Float64})
  if ! isempty(clamp)
    x[x .< clamp[1]] = clamp[1]
    x[x .> clamp[2]] = clamp[2]
  end
  return x
end

function ConditionalNormalWishart(U::Matrix{Float64}, mu::Vector{Float64}, beta_0::Real, Tinv::Matrix{Float64}, nu::Real)
  N  = size(U, 2)
  NU = sum(U, 2)
  NS = U * U'

  nu_N   = nu + N
  beta_N = beta_0 + N
  mu_N   = (beta_0*mu + NU) / (beta_0 + N)
  T_N    = inv( Tinv + NS + beta_0 * mu * mu' - beta_N * mu_N * mu_N')

  NormalWishart(vec(mu_N), beta_N, T_N, nu_N)
end

function sample_alpha(alpha_lambda0::Float64, alpha_nu0::Float64, err::Vector{Float64})
  Λ  = alpha_lambda0 * eye(1)
  n  = length(err)
  SW = inv(inv(Λ) + err' * err)
  return rand(Wishart(alpha_nu0 + n, SW))[1]
end

function sample_lambda_beta(β::Matrix{Float64}, Lambda_u::Matrix{Float64}, ν::Float64, μ::Float64)
  νx = ν + size(β, 1) * size(β, 2)
  μx = μ * νx / (ν + μ * trace( (β'*β) * Lambda_u) )
  b  = νx / 2
  c  = 2*μx / νx
  return rand(Gamma(b, c))
end

function grab_col{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, col::Integer)
  r = A.colptr[col]:A.colptr[col+1]-1
  A.rowval[r], A.nzval[r]
end

## sampling latent values for sample_u in parallel
function sample_latent_all!(sample_u::Matrix{Float64}, dataRefs::Vector, procs::Vector{Int}, mode::Int, mean_rating::Real, sample_m::Matrix{Float64}, alpha::Float64, mu_u, Lambda_u::Matrix{Float64})
  Nprocs = length(procs)
  length(dataRefs) == Nprocs || error("Number of procs ($(Nprocs)) must equal number of dataRefs($(length(dataRefs))).")

  ## 1. create split ranges 1:length(procs):idmax
  ranges = StepRange[ i:Nprocs:size(sample_u,2) for i in 1:Nprocs ]

  mu_vector = typeof(mu_u) <: Vector

  ## 2. call sample_latent_range at each worker with its dataRef
  @sync begin
    for i in 1:Nprocs
      @async begin
        if mu_vector
          sample_u[:, ranges[i]] = remotecall_fetch(procs[i], sample_latent_range_ref, ranges[i], dataRefs[i], mode, mean_rating, sample_m, alpha, mu_u, Lambda_u)
        else
          sample_u[:, ranges[i]] = remotecall_fetch(procs[i], sample_latent_range_ref, ranges[i], dataRefs[i], mode, mean_rating, sample_m, alpha, mu_u[:,ranges[i]], Lambda_u)
        end
      end
    end
  end
end

function sample_latent_range_ref(urange, Au_ref::RemoteRef, mode, mean_rating, sample_mt, alpha, mu_u, Lambda_u)
  Au = fetch(Au_ref)::FastIDF
  return sample_latent_range(urange, Au, mode, mean_rating, sample_mt, alpha, mu_u, Lambda_u)
end

## Sampling U, V for 2-way relation. Used by parallel code
## mu_u is Matrix of size num_latent x length(urange)
function sample_latent_range(urange, Au::FastIDF, mode::Int, mean_rating, sample_mt, alpha, mu_u::Matrix{Float64}, Lambda_u)
  U = zeros(size(mu_u, 1), length(urange))
  for i in 1:length(urange)
    u = urange[i]
    U[:,i] = sample_user_basic(u, Au, mode, mean_rating, sample_mt, alpha, mu_u[:,i], Lambda_u)
  end
  return U
end

## Sampling U, V for 2-way relation. Used by parallel code
function sample_latent_range(urange, Au::FastIDF, mode::Int, mean_rating, sample_mt, alpha, mu_u::Vector{Float64}, Lambda_u)
  U = zeros(length(mu_u), length(urange))
  for i in 1:length(urange)
    u = urange[i]
    U[:,i] = sample_user_basic(u, Au, mode, mean_rating, sample_mt, alpha, mu_u, Lambda_u)
  end
  return U
end

## uses sample_mt = sample_m'
function sample_user_basic(uu::Integer, Au::FastIDF, mode::Int, mean_rating, sample_mt::Matrix{Float64}, alpha::Float64, mu_u::Vector{Float64}, Lambda_u::Matrix{Float64})
  id, v = getData(Au, mode, uu)
  ff = id[:, mode == 1 ? 2 : 1]

  rr = v - mean_rating
  MM = sample_mt[:, ff]

  covar = inv(Lambda_u + alpha * (MM*MM'))
  mu    = covar * (alpha * MM * rr + Lambda_u * mu_u)

  # Sample from normal distribution
  chol(covar)' * randn(length(mu_u)) + mu
end

type Block
  ux::Vector{Int}   ## ids of the latent variables
  vx::Vector{Int}   ## ids of the other side
  Yma::Matrix{Float64} ## Y values w/o mean, size: length(v_idx) x length(u_idx)
end

function sample_users_blocked(block::Block, sample_mt::Matrix{Float64}, alpha::Float64, mu_u::Vector{Float64}, Lambda_u::Matrix{Float64})
  MM    = sample_mt[:, block.vx]
  covar = inv(Lambda_u + alpha * (MM*MM'))
  mu    = covar * (alpha * MM * block.Yma .+ Lambda_u * mu_u)

  # Sample from normal distribution
  chol(covar)' * randn(length(mu_u), size(mu, 2)) + mu
end

function sample_user2_all!(s::Entity, modes::Vector{Int64}, modes_other::Vector{Vector{Int64}})
  msample = s.model.sample
  mu      = s.model.mu
  for mm = 1:s.count
    msample[:, mm] = sample_user2(s, mm, mu, modes, modes_other)
  end
end

function sample_user2_all!(s::Entity, mu_matrix::Matrix{Float64}, modes::Vector{Int64}, modes_other::Vector{Vector{Int64}})
  msample = s.model.sample
  for mm = 1:s.count
    msample[:, mm] = sample_user2(s, mm, mu_matrix[:,mm], modes, modes_other)
  end
end

function sample_user2(s::Entity, i::Int, mu_si::Vector{Float64}, modes::Vector{Int64}, modes_other::Vector{Vector{Int64}})
  Lambda_si = copy(s.model.Lambda)
  mux       = s.model.Lambda * mu_si

  for r = 1:length(s.relations)
    rel = s.relations[r]
    df  = getData(rel.data, modes[r], i)
    rr  = convert(Array, df[:,end]) - (hasFeatures(rel) ? rel.temp.linear_values[getI(rel.data, modes[r], i)] : rel.model.mean_value)
    modes_o1 = modes_other[r][1]
    modes_o2 = modes_other[r][2:end]

    MM = rel.entities[modes_o1].model.sample[ :, df[:,modes_o1] ]
    for j = modes_o2
      MM .*= rel.entities[j].model.sample[ :, df[:,j] ]
    end
    Lambda_si += rel.model.alpha * MM * MM'
    mux       += rel.model.alpha * MM * rr
  end
  covar = inv(Lambda_si)
  mu    = covar * mux

  # Sample from normal distribution
  chol(covar)' * randn(length(mu)) + mu
end

function sample_beta(entity, sample_u_c, Lambda_u, lambda_beta, use_ff::Bool, tol=NaN )
  D, N = size(sample_u_c)
  numF = size(entity.F, 2)
  if isnan(tol)  ## default tolerance
    tol = eps() * numF
  end
  
  mv = MultivariateNormal(zeros(D), inv(Lambda_u) )
  ## TODO: using Ft_mul_Bt will be faster
  Ft_y = Ft_mul_B(entity, sample_u_c' + rand(mv, N)') + sqrt(lambda_beta) * rand(mv, numF)'
  #Ft_y = At_mul_B(entity.F, sample_u_c + rand(mv, N)') + sqrt(lambda_beta) * rand(mv, numF)'
  
  if use_ff
    beta = solve_full(entity.FF, Ft_y, lambda_beta)
  elseif ! isempty(entity.Frefs)
    beta = solve_cg2(entity.Frefs, Ft_y, lambda_beta, tol=tol)
  else
    error("No CG solver if entity has no Frefs or FF.")
    #beta = solve_cg(entity.F, Ft_y, lambda_beta)
  end
  return beta, Ft_y
end

function solve_full(FF, rhs, lambda_beta)
  FFreg = copy(FF)
  for i = 1:size(FFreg, 2)
    FFreg[i,i] += lambda_beta
  end
  return FFreg \ rhs
end

function sample_beta_rel(r::Relation)
  N, F = size(r.F)
  alpha  = r.model.alpha
  lambda = r.model.lambda_beta

  res  = getValues(r.data) - udot(r, r.data.df) - r.model.mean_value
  aFt_y = alpha * (r.F' * (res + alpha^(-0.5) * randn(N))) + sqrt(lambda) * randn(F)

  if isdefined(r.temp, :FF)
    ## using FF to compute beta_r
    K = alpha * r.temp.FF + lambda * speye(F)
    return K \ aFt_y
  else
    error("conjugate gradient unimplemented for sampling relation beta")
  end
end
