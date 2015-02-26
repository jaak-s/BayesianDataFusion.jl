include("sampling.jl")
include("purecg.jl")

export macau

function macau(data::RelationData;
              num_latent::Int = 10,
              lambda_beta     = NaN,
              burnin          = 500,
              psamples        = 200,
              verbose::Bool   = true,
              full_lambda_u   = false,
              reset_model     = true,
              clamp::Vector{Float64}  = Float64[],
              f::Union(Function,Bool) = false)
  correct = Float64[]

  if reset_model
    for en in data.entities
      initModel!(en, num_latent, lambda_beta = lambda_beta)
    end
  end

  modes = map(entity -> Int64[ find(en -> en == entity, r.entities)[1] for r in entity.relations ],
                     data.entities)
  modes_other = map(entity -> Vector{Int64}[ find(en -> en != entity, r.entities) for r in entity.relations ],
                     data.entities)

  verbose && println("Sampling")
  err_avg  = 0.0
  roc_avg  = 0.0
  rmse_avg = 0.0

  local probe_rat_all, clamped_rat_all
  f_output = Any[]

  ## Gibbs sampling loop
  for i in 1 : burnin + psamples
    time0 = time()

    # sample relation model (alpha)
    for r in data.relations
      r.model.alpha_sample || continue

      err = pred(r) - getValues(r.data)
      r.model.alpha = sample_alpha(r.model.alpha_lambda0, r.model.alpha_nu0, err)
    end

    rel = data.relations[1]

    # Sample from entity hyperparams
    for j in 1:length(data.entities)
      mj = data.entities[j].model

      local U::Matrix{Float64}
      local uhat::Matrix{Float64}
      nu   = num_latent
      Tinv = mj.WI

      if hasFeatures(data.entities[j])
        uhat = data.entities[j].F * mj.beta
        U = mj.sample - uhat
        if full_lambda_u
          nu   += size(mj.beta, 1)
          Tinv += mj.beta' * mj.beta * data.entities[j].lambda_beta
        end
      else
        U = mj.sample
      end

      mj.mu, mj.Lambda = rand( ConditionalNormalWishart(U, mj.mu0, mj.b0, Tinv, nu) )

      # latent vectors
      for mm = 1:data.entities[j].count
        mu_mm = hasFeatures(data.entities[j]) ? mj.mu + uhat[mm,:]' : mj.mu
        #mj.sample[mm, :] = sample_user(mm, rel.data, j, rel.mean_rating, data.entities[j2].model.sample, rel.model.alpha, mu_mm, mj.Lambda, num_latent)
        mj.sample[mm, :] = sample_user2(data.entities[j], mm, mu_mm, modes[j], modes_other[j])
      end

      if hasFeatures( data.entities[j] )
        mj.beta, rhs = sample_beta(data.entities[j].F, mj.sample .- mj.mu', mj.Lambda, data.entities[j].lambda_beta)
      end
    end

    ## TODO, use pred based on probe_vec and relation
    #probe_rat = pred(rel.test_vec, data.entities[2].model.sample, data.entities[1].model.sample, rel.mean_rating)
    probe_rat = pred(rel.test_vec, rel)

    if i > burnin
      if verbose && i == burnin + 1
        println("--------- Burn-in complete, averaging posterior samples ----------")
      end
      probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
      counter_prob  = counter_prob + 1
    else
      probe_rat_all = probe_rat
      counter_prob  = 1
    end

    if typeof(f) <: Function && i > burnin
      f_out = f( Matrix{Float64}[en.model.sample for en in data.entities] )
      push!(f_output, f_out)
    end

    time1    = time()
    haveTest = numTest(rel) > 0

    correct  = (rel.test_label .== (probe_rat_all .< rel.class_cut) )
    err_avg  = mean(correct)
    err      = mean(rel.test_label .== (probe_rat .< rel.class_cut))

    clamped_rat     = isempty(clamp) ?probe_rat     :makeClamped(probe_rat, clamp)
    clamped_rat_all = isempty(clamp) ?probe_rat_all :makeClamped(probe_rat_all, clamp)

    rmse_avg = haveTest ? sqrt(mean( (rel.test_vec[:,end] - clamped_rat_all) .^ 2 )) : NaN
    rmse     = haveTest ? sqrt(mean( (rel.test_vec[:,end] - clamped_rat) .^ 2 ))     : NaN
    roc_avg  = haveTest ? AUC_ROC(rel.test_label, -vec(probe_rat_all))             : NaN
    verbose && @printf("Iter %3d: Acc %6.4f | avgAcc %6.4f avgROC %6.4f avgRMSE %6.4f | FU(%6.2f) FM(%6.2f) Fb(%6.2f) α=%2.1f [%2.0fs]\n", i, err, err_avg, roc_avg, rmse_avg, vecnorm(data.entities[1].model.sample), vecnorm(data.entities[2].model.sample), vecnorm(data.entities[1].model.beta), data.relations[1].model.alpha, time1 - time0)
  end

  result = Dict{String,Any}()
  result["num_latent"]  = num_latent
  result["burnin"]      = burnin
  result["psamples"]    = psamples
  result["lambda_beta"] = data.entities[1].lambda_beta
  result["RMSE"]        = rmse_avg
  result["accuracy"]    = err_avg
  result["ROC"]         = roc_avg
  if numTest(data.relations[1]) > 0
    rel = data.relations[1]
    result["predictions"] = copy(rel.test_vec)
    result["predictions"][:pred] = vec(clamped_rat_all)
    train_count = zeros(Int, numTest(rel), length(size(rel)) )
    for i in 1:numTest(rel)
      for mode in 1:length(size(rel))
        train_count[i,mode] = getCount(rel.data, mode, rel.test_vec[i,mode])
      end
    end
    result["train_counts"] = convert( DataFrame, train_count )
  end
  if typeof(f) <: Function
    result["f_output"] = f_output
  end
  return result
end
