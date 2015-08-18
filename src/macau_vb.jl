export bpmf_vb
export VBModel

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
    randn(num_latent, N),              ## mu_u
    zeros(num_latent, num_latent, N),  ## Euu
    num_latent + N,                    ## nu_N
    eye(num_latent, num_latent),       ## W_N
    zeros(num_latent),                 ## mu_N
    2.0 + N,                           ## b_N (beta)
    eye(num_latent, num_latent),       ## Winv_0
    zeros(num_latent),                 ## mu_0
    2.0                                ## b_0
  )
  for n in 1:N
    m.Euu[:,:,n] = eye(num_latent)
  end
  return m
end

function bpmf_vb(data::RelationData,
                 num_latent::Int = 10,
                 verbose::Bool   = true,
                 niter::Int      = 100)
  ## initialization
  Umodel = VBModel(num_latent, data.entities[1].count)
  Vmodel = VBModel(num_latent, data.entities[2].count)

  ## data
  df  = data.relations[1].data.df
  uid = convert(Vector{Int32}, df[:,1])
  vid = convert(Vector{Int32}, df[:,2])
  val = convert(Array, df[:,3])
  Udata = sparse(vid, uid, val)
  Vdata = sparse(uid, vid, val)

  alpha = data.relations[1].model.alpha

  update_u!(Umodel, Vmodel, Udata, alpha)
  update_u!(Vmodel, Umodel, Vdata, alpha)

  update_prior!(Umodel)
  update_prior!(Vmodel)
end

function update_u!(Umodel::VBModel, Vmodel::VBModel, data::SparseMatrixCSC, alpha::Float64)
  Ex_L_U = Umodel.W_N * Umodel.nu_N
  Ex_Lmu = Umodel.W_N * Umodel.nu_N * Umodel.mu_N
  num_latent = size(Umodel.mu_u, 1)

  for uu in 1:size(Umodel.mu_u, 2)
    L  = copy(Ex_L_U)
    av = copy(Ex_Lmu)
    for j in data.colptr[uu] : data.colptr[uu+1]-1
      vv  = data.rowval[ j ]
      L  += alpha * Vmodel.Euu[:, :, vv]
      av += alpha * data.nzval[j] * Vmodel.mu_u[:, vv]
    end
    Linv = inv(L)
    Umodel.mu_u[:, uu] = Linv * av
    Umodel.Euu[:,:,uu] = Linv + av * av'
  end
  nothing
end

function update_prior!(m::VBModel)
  num_latent = size(m.mu_u, 1)

  m.mu_N = (m.b_0 * m.mu_0 + vec(sum(m.mu_u, 2)) ) / m.b_N
  m.W_N  = m.Winv_0 + reshape(sum(m.Euu, 3), num_latent, num_latent)
  m.W_N += m.b_0 * m.mu_0 * m.mu_0' - m.b_N * m.mu_N * m.mu_N'
  m.W_N = inv(m.W_N)
  nothing
end
