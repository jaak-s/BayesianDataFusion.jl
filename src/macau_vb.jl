export bpmf_vb
export VBModel

using Compat

type VBModel
  mu_u::Matrix{Float64}    ## means of u_i
  Euu ::Array{Float64, 3}  ## storing E[u u'] = Lambda_u^{-1} + mu_u mu_u'

  nu_N::Float64            ## nu for Normal-Wishart
  W_N ::Matrix{Float64}    ## W  for Normal-Wishart
  mu_N::Vector{Float64}    ## mu for Normal-Wishart
  b_N ::Float64            ## beta for Normal-Wishart

  Winv_0::Matrix{Float64}  ## hyperprior W_0 inverted
  mu_0::Vector{Float64}    ## hyperprior mu_0
  b_0::Float64             ## hyperprior b_0
end

function VBModel(num_latent::Int, N::Int)
  m = VBModel(
    1.0*randn(num_latent, N),          ## mu_u
    zeros(num_latent, num_latent, N),  ## Euu
    num_latent + N,                    ## nu_N
    1/N*eye(num_latent, num_latent),   ## W_N
    zeros(num_latent),                 ## mu_N
    2.0 + N,                           ## b_N (beta)
    eye(num_latent, num_latent),       ## Winv_0
    zeros(num_latent),                 ## mu_0
    2.0                                ## b_0
  )
  for n in 1:N
    ## E[u u'] = inv(L_u_i) + mu_u * mu_u'
    m.Euu[:,:,n] = inv(m.W_N) + m.mu_u[:,n] * m.mu_u[:,n]'
  end
  return m
end

function bpmf_vb(data::RelationData;
                 num_latent::Int = 10,
                 verbose::Bool   = true,
                 niter::Int      = 100,
                 clamp::Vector{Float64} = Float64[])
  ## initialization
  rmse = NaN
  rmse_train = NaN
  Umodel = VBModel(num_latent, data.entities[1].count)
  Vmodel = VBModel(num_latent, data.entities[2].count)

  ## data
  df  = data.relations[1].data.df
  mean_value = mean(df[:,end])
  uid = convert(Vector{Int32}, df[:,1])
  vid = convert(Vector{Int32}, df[:,2])
  val = convert(Array, df[:,3]) - mean_value
  Udata = sparse(vid, uid, val)
  Vdata = sparse(uid, vid, val)

  test_uid = convert(Array, data.relations[1].test_vec[:,1])
  test_vid = convert(Array, data.relations[1].test_vec[:,2])
  test_val = convert(Array, data.relations[1].test_vec[:,3])

  alpha = data.relations[1].model.alpha

  for i in 1:niter
    update_u!(Umodel, Vmodel, Udata, alpha)
    update_u!(Vmodel, Umodel, Vdata, alpha)

    update_prior!(Umodel)
    update_prior!(Vmodel)

    if verbose
      yhat = clamp!(predict(Umodel, Vmodel, mean_value, test_uid, test_vid), clamp)
      rmse = sqrt( mean((yhat - test_val).^2) )
      yhat_train = clamp!(predict(Umodel, Vmodel, mean_value, uid, vid), clamp)
      rmse_train = sqrt( mean((yhat_train - mean_value - val).^2) )
      @printf("% 3d: |U|=%.4e  |V|=%.4e  RMSE=%.4f  RMSE(train)=%.4f\n", i, vecnorm(Umodel.mu_u), vecnorm(Vmodel.mu_u), rmse, rmse_train)
    end
  end
  return @compat Dict(
    "Umodel" => Umodel,
    "Vmodel" => Vmodel,
    "rmse"   => rmse,
    "rmse_train" => rmse_train)
end

function update_u!(Umodel::VBModel, Vmodel::VBModel, Udata::SparseMatrixCSC, alpha::Float64)
  A = Umodel.W_N * Umodel.nu_N
  b = Umodel.W_N * Umodel.nu_N * Umodel.mu_N
  num_latent = size(Umodel.mu_u, 1)
  colptr = Udata.colptr
  rowval = Udata.rowval
  nzval  = Udata.nzval

  for uu in 1:size(Umodel.mu_u, 2)
    L  = copy(A)
    av = copy(b)
    for j in colptr[uu] : colptr[uu+1]-1
      vv  = rowval[ j ]
      L  += alpha * Vmodel.Euu[:, :, vv]
      av += alpha * nzval[j] * Vmodel.mu_u[:, vv]
    end
    Linv = inv(L)
    mu   = Linv * av
    Umodel.mu_u[:, uu] = mu
    Umodel.Euu[:,:,uu] = Linv + mu * mu'
  end
  nothing
end

function update_prior!(m::VBModel)
  num_latent = size(m.mu_u, 1)

  m.mu_N = (m.b_0 * m.mu_0 + vec(sum(m.mu_u, 2)) ) / m.b_N

  m.W_N  = inv(m.Winv_0 + reshape(sum(m.Euu, 3), num_latent, num_latent)
               + m.b_0 * m.mu_0 * m.mu_0' - m.b_N * m.mu_N * m.mu_N' )
  nothing
end

function predict(Umodel, Vmodel, mean_value, uids::Vector, vids::Vector)
  yhat = zeros(length(uids)) + mean_value
  for i in 1:length(uids)
    yhat[i] += dot( Umodel.mu_u[:,uids[i]], Vmodel.mu_u[:,vids[i]] )
  end
  return yhat
end

import Base.show
function show(io::IO, m::VBModel)
  @printf(io, "VBModel of %d instances: |mu_u|=%0.3e", size(m.mu_u, 2), vecnorm(m.mu_u))
end
