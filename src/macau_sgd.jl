export macau_sgd

function macau_sgd(data::RelationData;
                   num_latent::Int = 10,
                   verbose::Bool   = true,
                   niter::Int      = 100,
                   lrate::Float64  = 0.001,
                   reset_model     = true,
                   batch_size      = 4000,
                   clamp::Vector{Float64} = Float64[])
  ## initialization
  verbose     && println("Model setup")
  reset_model && reset!(data, num_latent)

  rmse = NaN
  rmse_sample = NaN
  rmse_train  = NaN
  #Umodel = HMCModel(num_latent, data.entities[1].count, diag(data.entities[1].model.Lambda))
  #Vmodel = HMCModel(num_latent, data.entities[2].count, diag(data.entities[2].model.Lambda))

  ## data
  df  = data.relations[1].data.df
  mean_value = mean(df[:,end])
  uid = convert(Vector{Int32}, df[:,1])
  vid = convert(Vector{Int32}, df[:,2])
  val = convert(Array, df[:,3]) - mean_value

  Ubatches = create_minibatches(vid, uid, val, batch_size)
  Vbatches = [x' for x in Ubatches]

  Usample  = data.entities[1].model.sample
  Vsample  = data.entities[2].model.sample

  test_uid = convert(Vector{Int}, data.relations[1].test_vec[:,1])
  test_vid = convert(Vector{Int}, data.relations[1].test_vec[:,2])
  test_val = convert(Vector, data.relations[1].test_vec[:,3])
  test_idx = hcat(test_uid, test_vid)

  alpha = data.relations[1].model.alpha
  yhat_post = zeros(length(test_val))

  @time for i in 1:length(Ubatches)
    sgd_update!(Usample, Vsample, Ubatches[i], data.entities[1].model, alpha)
    sgd_update!(Vsample, Usample, Vbatches[i], data.entities[2].model, alpha)
  end
  return Ubatches, Vbatches

end

function create_minibatches{Ti,Tv}(uid::Vector{Ti}, vid::Vector{Ti}, val::Vector{Tv}, bsize::Int)
  perm = randperm(length(uid))
  batches = SparseMatrixCSC{Tv,Ti}[]
  umax = maximum(uid)
  vmax = maximum(vid)
  
  for i in 1:bsize:length(uid)
    j = min(length(uid), i + bsize - 1)
    push!(batches, sparse(uid[perm[i:j]], vid[perm[i:j]], val[perm[i:j]], umax, vmax) )
  end
  return batches
end

function sgd_update!(sample, Vsample, batch::SparseMatrixCSC, model::EntityModel, alpha)
  for n in 1:size(sample, 2)
    grad!(n, sample, Vsample, batch, model.Lambda, model.mu, alpha)
  end
  nothing
end

function grad!(n,
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

  sample[:,n] += alpha * (MM * rr - (MM*(MM' * un))) + Lambda * (mu - un)
  nothing
end

function grad_yhat(
              U       ::Matrix{Float64},
              V       ::Matrix{Float64},
              colptr  ::Vector,
              rowval  ::Vector,
              #nzval   ::Vector,
              resid   ::Vector,
              Lambda  ::Matrix{Float64},
              mu      ::Vector{Float64},
              alpha   ::Float64)
  num_latent = size(U, 1)
  grad = Lambda * (mu .- U)
  @inbounds for n in 1:size(U,2)
    @inbounds for i in colptr[n] : colptr[n+1]-1
      err_i = alpha * resid[i]
      v_i   = rowval[i]
      @inbounds @simd for k in 1:num_latent
        grad[k,n] += err_i * V[k, v_i]
      end
    end
  end

  return grad
end

function grad_yhat!(
              grad    ::Matrix{Float64},
              U       ::Matrix{Float64},
              V       ::Matrix{Float64},
              colptr  ::Vector,
              rowval  ::Vector,
              #nzval   ::Vector,
              resid   ::Vector,
              lambda  ::Vector{Float64},
              mu      ::Vector{Float64},
              alpha   ::Float64)
  num_latent = size(U, 1)
  @inbounds for n in 1:size(U,2)
    @inbounds @simd for k in 1:num_latent
      grad[n, k] = (U[n, k] - mu[k]) * lambda[k]
    end
  end
  @inbounds for n in 1:size(U,2)
    @inbounds for i in colptr[n] : colptr[n+1]-1
      err_i = alpha * resid[i]
      v_i   = rowval[i]
      @inbounds @simd for k in 1:num_latent
        grad[k,n] += err_i * V[k, v_i]
      end
    end
  end

  return grad
end
