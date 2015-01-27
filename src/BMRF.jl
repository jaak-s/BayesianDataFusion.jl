include("sampling.jl")
include("purecg.jl")

function BMRF(data::RelationData;
              num_latent::Int = 10,
              lambda_beta     = 1.0,
              alpha           = 5.0,
              burnin        = 500,
              psamples      = 200,
              class_cut     = log10(200),
              verbose::Bool = true,
              clamp::Vector{Float64} = Float64[])
  correct = Float64[] 

  initModel!(data.entities[1], num_latent, lambda_beta = lambda_beta)
  initModel!(data.entities[2], num_latent, lambda_beta = lambda_beta)

  verbose && println("Sampling")
  err_avg  = 0.0
  roc_avg  = 0.0
  rmse_avg = 0.0

  local probe_rat_all

  ## Gibbs sampling loop
  for i in 1 : burnin + psamples
    time0 = time()

    rel = data.relations[1]

    # Sample from movie hyperparams
    for j in 1:length(data.entities)
      j2 = (j == 1) ? 2 : 1
      mj = data.entities[j].model

      local U::Matrix{Float64}
      local uhat::Matrix{Float64}

      if hasFeatures(data.entities[j])
        uhat = data.entities[j].F * mj.beta
        U = mj.sample - uhat
      else
        U = mj.sample
      end

      mj.mu, mj.Lambda = rand( ConditionalNormalWishart(U, mj.mu0, mj.b0, mj.WI, num_latent) )

      # latent vectors
      for mm = 1:data.entities[j].count
        mu_mm = hasFeatures(data.entities[j]) ? mj.mu + uhat[mm,:]' : mj.mu
        mj.sample[mm, :] = sample_user(mm, rel.data, j, rel.mean_rating, data.entities[j2].model.sample, alpha, mu_mm, mj.Lambda, num_latent)
      end

      if hasFeatures( data.entities[j] )
        mj.beta, rhs = sample_beta(data.entities[j].F, mj.sample .- mj.mu', mj.Lambda, mj.lambda_beta)
      end
    end

    # clamping maybe needed for MovieLens data
    local probe_rat
    if isempty(clamp)
      probe_rat = pred(rel.test_vec, data.entities[2].model.sample, data.entities[1].model.sample, rel.mean_rating)
    else
      probe_rat = pred(rel.test_vec, data.entities[2].model.sample, data.entities[1].model.sample, rel.mean_rating)
      probe_rat[ probe_rat .< clamp[1] ] = clamp[1]
      probe_rat[ probe_rat .> clamp[2] ] = clamp[2]
    end

    if i > burnin
      probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
      counter_prob  = counter_prob + 1
    else
      probe_rat_all = probe_rat
      counter_prob  = 1
    end

    time1    = time()
    haveTest = size(rel.test_label,1) > 0

    correct  = (rel.test_label .== (probe_rat_all .< class_cut) )
    err_avg  = mean(correct)
    err      = mean(rel.test_label .== (probe_rat .< class_cut))
    rmse_avg = haveTest ? sqrt(mean( (rel.test_vec[:,3] - probe_rat_all) .^ 2 )) : NaN
    rmse     = haveTest ? sqrt(mean( (rel.test_vec[:,3] - probe_rat) .^ 2 ))     : NaN
    roc_avg  = haveTest ? AUC_ROC(rel.test_label, -vec(probe_rat_all))           : NaN
    verbose && @printf("Iteration %d:\t avgAcc %6.4f Acc %6.4f | avgRMSE %6.4f | avgROC %6.4f | FU(%6.2f) FM(%6.2f) Fb(%6.2f) [%2.0fs]\n", i, err_avg, err, rmse_avg, roc_avg, vecnorm(data.entities[1].model.sample), vecnorm(data.entities[2].model.sample), vecnorm(data.entities[1].model.beta), time1 - time0)
  end

  result = Dict{String,Any}()
  result["num_latent"]  = num_latent
  result["RMSE"]        = rmse_avg
  result["accuracy"]    = err_avg
  result["ROC"]         = roc_avg
  result["predictions"] = probe_rat_all
  return result
end
