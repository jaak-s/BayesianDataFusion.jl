export macau_hmc

using Compat

type HMCModel
  momentum::Matrix{Float64}
  G       ::Matrix{Float64} ## diagonal mass matrix
end

deepcopy(m::HMCModel) = HMCModel(copy(m.momentum))

function HMCModel(num_latent::Int, N::Int, Ldiag::Vector{Float64})
  return HMCModel(
    zeros(num_latent, N),  ## momentum
    repmat(Ldiag, 1, N)    ## mass matrix (G)
  )
end

function macau_hmc(data::RelationData;
                   num_latent::Int = 10,
                   verbose::Bool   = true,
                   burnin::Int     = 100,
                   psamples::Int   = 100,
                   L::Int          = 10,
                   L_inner::Int    = 1,
                   prior_freq::Int = 8,  ## how often to update prior
                   eps::Float64    = 0.01,
                   reset_model     = true,
                   clamp::Vector{Float64} = Float64[])
  ## initialization
  verbose     && println("Model setup")
  reset_model && reset!(data, num_latent)

  rmse = NaN
  rmse_sample = NaN
  rmse_train  = NaN
  Umodel = HMCModel(num_latent, data.entities[1].count, diag(data.entities[1].model.Lambda))
  Vmodel = HMCModel(num_latent, data.entities[2].count, diag(data.entities[2].model.Lambda))


  ## data
  df  = data.relations[1].data.df
  mean_value = mean(df[:,end])
  uid = convert(Vector{Int32}, df[:,1])
  vid = convert(Vector{Int32}, df[:,2])
  val = convert(Array, df[:,3]) - mean_value
  Udata = sparse(vid, uid, val, data.entities[2].count, data.entities[1].count)
  Vdata = Udata'

  test_uid = convert(Vector{Int}, data.relations[1].test_vec[:,1])
  test_vid = convert(Vector{Int}, data.relations[1].test_vec[:,2])
  test_val = convert(Vector, data.relations[1].test_vec[:,3])
  test_idx = hcat(test_uid, test_vid)

  alpha = data.relations[1].model.alpha
  yhat_post = zeros(length(test_val))

  for i in 1 : burnin + psamples
    time0 = time()

    if i == burnin + 1
      verbose && print("================== Burnin complete ===================\n")
    end

    verbose && @printf("======= Step %d =======\n", i)
    verbose && @printf("eps = %.2e\n", eps)

    # HMC sampling momentum
    sample!(Umodel)
    sample!(Vmodel)
    kinetic_start   = computeKinetic(Umodel) + computeKinetic(Vmodel)
    potential_start = computePotential(uid, vid, val, data.relations[1])
    Ustart = copy(data.entities[1].model.sample)
    Vstart = copy(data.entities[2].model.sample)

    # Follow Hamiltonians
    hmc_update_u!(Umodel, Vmodel, Udata, alpha, data.entities[1], data.entities[2], L_inner, eps / 2)
    for l in 1:L
      hmc_update_u!(Vmodel, Umodel, Vdata, alpha, data.entities[2], data.entities[1], L_inner, eps)
      if l < L
        hmc_update_u!(Umodel, Vmodel, Udata, alpha, data.entities[1], data.entities[2], L_inner, eps)
      end
      verbose && @printf("  Momentum %d: |r_U| = %.4e, |r_V| = %.4e\n", l, vecnorm(Umodel.momentum), vecnorm(Vmodel.momentum))
    end
    hmc_update_u!(Umodel, Vmodel, Udata, alpha, data.entities[1], data.entities[2], L_inner, eps / 2)
    verbose && @printf("  Momentum L: |r_U| = %.4e, |r_V| = %.4e\n", vecnorm(Umodel.momentum), vecnorm(Vmodel.momentum))


    # Verify p(U,V) is fine
    kinetic_final   = computeKinetic(Umodel) + computeKinetic(Vmodel)
    potential_final = computePotential(uid, vid, val, data.relations[1])

    dH = potential_start - potential_final + kinetic_start - kinetic_final
    verbose && @printf("  ΔH = %.4e  ΔKin = %.4e  ΔPot = %.4e\n",
                       -dH,
                       kinetic_final-kinetic_start, potential_final - potential_start)
    if rand() < exp(dH)
      ## accept
      verbose && print("-> ACCEPTED!\n")
    else
      ## reject
      verbose && print("-> REJECTED!\n")
      Base.copy!(data.entities[1].model.sample, Ustart)
      Base.copy!(data.entities[2].model.sample, Vstart)
      if dH < -6
        ## decreasing eps by 2
        neweps = eps / 2
        newL   = ceil(Int, L*1.6)
        verbose && @printf("Reducing eps from %.2e to %.2e.\n", eps, neweps)
        verbose && @printf("Increasing L from %d to %d.\n", L, newL)
        eps = neweps
        L   = newL
      end
    end

    # Sampling prior for latents
    if i % prior_freq == 0
      verbose && println("Updating priors...")
      for en in data.entities
        update_latent_prior!(en, true)
      end
    end

    rel  = data.relations[1]
    yhat_raw = pred(rel, test_idx, rel.test_F)
    yhat     = clamp!(yhat_raw, clamp)
    update_yhat_post!(yhat_post, yhat_raw, i, burnin)

    rmse = sqrt( mean((yhat - test_val).^2) )
    rmse_post = sqrt( mean((makeClamped(yhat_post, clamp) - test_val).^2) )
    #yhat_train = clamp!(pred(Umodel, Vmodel, mean_value, uid, vid), clamp)
    #rmse_train = sqrt( mean((yhat_train - mean_value - val).^2) )
    time1 = time()


    if verbose
      @printf("% 3d: |U|=%.4e  |V|=%.4e  RMSE=%.4f  RMSE(avg)=%.4f [took %.2fs]\n",
        i,
        vecnorm(data.entities[1].model.sample),
        vecnorm(data.entities[2].model.sample),
        rmse,
        rmse_post,
        time1 - time0)
    end
  end

  return @compat Dict(
    "rmse"   => rmse,
    "rmse_train" => rmse_train,
    "alpha"  => alpha)
end

function sample!(m::HMCModel)
  ## sampling from Normal distribution with identity cov
  for i = 1:size(m.momentum,1), j = 1:size(m.momentum,2)
    m.momentum[i,j] = randn() / sqrt(m.G[i,j])
  end
  nothing
end


function hmc_update_u!(U_hmcmodel::HMCModel,
                       V_hmcModel::HMCModel,
                       Udata,
                       alpha::Float64,
                       en::Entity,
                       enV::Entity,
                       L::Int,
                       eps::Float64)

  model      = en.model
  sample     = model.sample
  Vsample    = enV.model.sample
  momentum   = U_hmcmodel.momentum
  num_latent = size(U_hmcmodel.momentum, 1)

  # 1) make half step in momentum
  subtract_grad!(momentum, model, sample, Vsample, Udata, alpha, eps / 2)

  # 2) L times: make full steps with U, full step with momentum (except for last step)
  for i in 1:L
    add!(sample, momentum, eps)
    if i < L
      subtract_grad!(momentum, model, sample, Vsample, Udata, alpha, eps)
    end
  end

  # 3) make half step in momentum
  subtract_grad!(momentum, model, sample, Vsample, Udata, alpha, eps / 2)
  nothing
end

## computes X += mult * Y
function add!(X::Matrix, Y::Matrix, mult::Float64)
  size(X) == size(Y) || error("X and Y must have the same size.")
  @inbounds @simd for i in 1:length(X)
    X[i] += mult * Y[i]
  end
  nothing
end

function subtract_grad!(momentum::Matrix{Float64},
                        model   ::EntityModel,
                        sample  ::Matrix{Float64},
                        Vsample ::Matrix{Float64},
                        Udata   ::SparseMatrixCSC,
                        alpha   ::Float64,
                        eps     ::Float64)
  num_latent = size(momentum, 1)

  for n in 1:size(sample, 2)
    tmp = grad(n, sample, Vsample, Udata, model.Lambda, model.mu, alpha)
    @inbounds for k in 1:num_latent
      momentum[k,n] -= eps * tmp[k]
    end
  end
  nothing
end


function grad(n,
              sample  ::Matrix{Float64},
              Vsample ::Matrix{Float64},
              Udata   ::SparseMatrixCSC,
              Lambda  ::Matrix{Float64},
              mu      ::Vector{Float64},
              alpha   ::Float64)
  un  = sample[:, n]
  idx = Udata.colptr[n] : Udata.colptr[n+1]-1
  ff  = Udata.rowval[ idx ]
  rr  = Udata.nzval[ idx ]

  MM = Vsample[:, ff]
  ## add d/du A(r_V | U)
  ## add d/du 1/2 log|G_V|

  return - alpha * (MM * rr - (MM*MM') * un) - Lambda * (mu - un)
end

function computeKinetic(m::HMCModel)
  kin = 0.0
  momentum = m.momentum
  G        = m.G
  ## r'*G*r + log|G|
  @inbounds @simd for i = 1:length(momentum)
    kin += momentum[i] * momentum[i] * G[i] + log(G[i])
  end
  return 0.5 * kin
end

## computes - log [p(Y | U,V) p(U, V)]
function computePotential(uid::Vector, vid::Vector, val::Vector, r::Relation)
  alpha::Float64 = r.model.alpha
  Umodel = r.entities[1].model
  Vmodel = r.entities[2].model
  Usample = Umodel.sample
  Vsample = Vmodel.sample
  energy  = 0.0

  ## data
  for i in 1:length(uid)
    energy += (column_dot(Usample, Vsample, uid[i], vid[i]) - val[i]) ^ 2
  end
  energy *= alpha / 2

  ## Priors
  ## sum(u_i' * Lambda_u * u_i) = trace(Lambda_u * U * U')
  energy += sum(Umodel.Lambda .* (Usample * Usample')) / 2
  energy += sum(Vmodel.Lambda .* (Vsample * Vsample')) / 2

  ## -2 * sum(mu_u' * Lambda_u * u_i)
  energy -= first(Umodel.mu' * Umodel.Lambda * sum(Usample, 2))
  energy -= first(Vmodel.mu' * Vmodel.Lambda * sum(Vsample, 2))

  return energy
end

function column_dot(X, Y, i, j)
  size(X,1) == size(Y,1) || error("X and Y must have the same number of rows.")
  size(X,2) >= i         || error("X must have at least i columns.")
  size(Y,2) >= j         || error("Y must have at least j columns.")
  d = 0.0
  @inbounds for k = 1:size(X,1)
    d += X[k,i] * Y[k,j]
  end
  return d
end

function update_yhat_post!(yhat_post, yhat_raw, i, burnin)
  ## also takes care of the first sample of posterior
  if i <= burnin + 1
    Base.copy!(yhat_post, yhat_raw)
    return yhat_post
  end

  ## averaging
  n = i - burnin - 1
  Base.copy!(yhat_post, (n*yhat_post + yhat_raw) / (n + 1))
  return yhat_post
end
