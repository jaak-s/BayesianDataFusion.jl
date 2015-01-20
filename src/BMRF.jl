include("sampling.jl")
include("purecg.jl")

function BMRF(data::RelationData; num_latent::Int = 10, lambda_beta = 1.0, alpha = 5.0, burnin = 500, psamples = 100, class_cut = log10(200), verbose::Bool = true)
  correct = Float64[] 

  sample_u = zeros(data.num_p, num_latent)
  sample_m = zeros(data.num_m, num_latent)

  # Initialize hierarchical priors
  mu_u = zeros(num_latent, 1)
  mu_m = zeros(num_latent, 1)
  Lambda_u = eye(num_latent)
  Lambda_m = eye(num_latent)

  # parameters of Inv-Whishart distribution (see paper for details)
  WI_u = eye(num_latent)
  b0_u = 2.0
  df_u = num_latent
  mu0_u = zeros(num_latent, 1)

  WI_m = eye(num_latent)
  b0_m = 2.0
  df_m = num_latent
  mu0_m = zeros(num_latent,1)

  # BMRF feature parameter
  beta = zeros(size(data.F, 2), num_latent)

  verbose && println("Sampling")
  err_avg  = 0.0
  roc_avg  = 0.0
  rmse_avg = 0.0

  local probe_rat_all

  ## Gibbs sampling loop
  for i in 1 : burnin + psamples
    time0 = time()
    # Sample from movie hyperparams
    mu_m, Lambda_m = rand( ConditionalNormalWishart(sample_m, vec(mu0_m), b0_m, WI_m, df_m) )

    # Sample from user hyperparams
    # for BMRF using U - data.F * beta (residual) instead of U
    uhat = data.F * beta
    mu_u, Lambda_u = rand( ConditionalNormalWishart(sample_u - uhat, vec(mu0_u), b0_u, WI_u, df_u) )

    for mm = 1:data.num_m
      sample_m[mm, :] = sample_movie(mm, data.Am, data.mean_rating, sample_u, alpha, mu_m, Lambda_m, num_latent)
    end

    # BMRF, instead of mu_u using mu_u + data.F * beta    
    for uu = 1:data.num_p
      sample_u[uu, :] = sample_user(uu, data.Au, data.mean_rating, sample_m, alpha, mu_u + uhat[uu,:]', Lambda_u, num_latent)
    end
    
    # sampling beta (using GAMBL-R trick)
    beta = sample_beta(data.F, data.Ft, sample_u .- mu_u', Lambda_u, lambda_beta)

    # clamping maybe needed for MovieLens data
    probe_rat = pred(data.probe_vec, sample_m, sample_u, data.mean_rating)
    #else
    #  probe_rat = pred_clamp(probe_vec, sample_m, sample_u, mean_rating)
    #end

    if i > burnin
      probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
      counter_prob  = counter_prob + 1
    else
      probe_rat_all = probe_rat
      counter_prob  = 1
    end

    time1    = time()
    correct  = (data.ratings_test .== (probe_rat_all .< class_cut) )
    err_avg  = mean(correct)
    err      = mean(data.ratings_test .== (probe_rat .< class_cut))
    rmse_avg = sqrt(mean( (data.probe_vec[:,3] - probe_rat_all) .^ 2 ))
    rmse     = sqrt(mean( (data.probe_vec[:,3] - probe_rat) .^ 2 ))
    roc_avg  = AUC_ROC(data.ratings_test, -vec(probe_rat_all))
    verbose && @printf("Iteration %d:\t avgAcc %6.4f Acc %6.4f | avgRMSE %6.4f | avgROC %6.4f | FU(%6.2f) FM(%6.2f) Fb(%6.2f) [%2.0fs]\n", i, err_avg, err, rmse_avg, roc_avg, vecnorm(sample_u), vecnorm(sample_m), vecnorm(beta), time1 - time0)

  end
  result = Dict{String,Any}()
  result["num_latent"]  = num_latent
  result["RMSE"]        = rmse_avg
  result["accuracy"]    = err_avg
  result["ROC"]         = roc_avg
  result["predictions"] = probe_rat_all
  return result
end
