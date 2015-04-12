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
              compute_ff_size = 6500,
              full_prediction = false,
              clamp::Vector{Float64}  = Float64[],
              f::Union(Function,Bool) = false)
  correct = Float64[]
  local yhat_full

  verbose && println("Model setup")

  if reset_model
    for en in data.entities
      initModel!(en, num_latent, lambda_beta = lambda_beta)
      if hasFeatures(en) && size(en.F,2) <= compute_ff_size
        en.FF = full(At_mul_B(en.F, en.F))
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

  modes = map(entity -> Int64[ find(en -> en == entity, r.entities)[1] for r in entity.relations ],
                     data.entities)
  modes_other = map(entity -> Vector{Int64}[ find(en -> en != entity, r.entities) for r in entity.relations ],
                     data.entities)
  if full_prediction
    yhat_full = zeros(size(data.relations[1]))
  end

  verbose && println("Sampling")
  err_avg  = 0.0
  roc_avg  = 0.0
  rmse_avg = 0.0

  local probe_rat_all, clamped_rat_all
  local train_rat_all
  f_output = Any[]

  ## Gibbs sampling loop
  for i in 1 : burnin + psamples
    time0 = time()

    # sample relation model (alpha)
    for r in data.relations
      if r.model.alpha_sample
        err = pred(r) - getValues(r.data)
        r.model.alpha = sample_alpha(r.model.alpha_lambda0, r.model.alpha_nu0, err)
      end
      if hasFeatures(r)
        r.model.beta = sample_beta_rel(r)
        r.temp.linear_values = r.model.mean_value + r.F * r.model.beta
      end
    end

    # Sample from entity hyperparams
    for j in 1:length(data.entities)
      en = data.entities[j]
      mj = en.model

      local U::Matrix{Float64}
      local uhat::Matrix{Float64}
      nu   = num_latent
      Tinv = mj.WI

      if hasFeatures(en)
        uhat = en.F * mj.beta
        U = mj.sample - uhat
        if full_lambda_u
          nu   += size(mj.beta, 1)
          Tinv += mj.beta' * mj.beta * en.lambda_beta
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
        use_ff = size(data.entities[j].F, 2) <= compute_ff_size
        mj.beta, rhs = sample_beta(data.entities[j], mj.sample .- mj.mu', mj.Lambda, data.entities[j].lambda_beta, use_ff)
        if en.lambda_beta_sample
          en.lambda_beta = sample_lambda_beta(mj.beta, mj.Lambda, en.nu, en.mu)
        end
      end
    end

    rel = data.relations[1]
    probe_rat = pred(rel, rel.test_vec, rel.test_F)
    train_rat = pred(rel)

    if full_prediction && i > burnin
      yhat_full += pred_all( data.relations[1] )
    end

    if i > burnin
      train_rat = pred(rel)
      if i == burnin + 1
        verbose && println("--------- Burn-in complete, averaging posterior samples ----------")
        counter_prob  = 1
        probe_rat_all = probe_rat
        train_rat_all = train_rat
      else
        probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
        train_rat_all = (counter_prob*train_rat_all + train_rat)/(counter_prob+1)
        counter_prob  = counter_prob + 1
      end
    else
      probe_rat_all = probe_rat
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
    if verbose
      estr = join(map(en -> toStr(en), data.entities), " ")
      rstr = join(map(r  -> toStr(r),  data.relations), " ")
      if i <= burnin
        verbose && @printf("%3d: Acc=%6.4f ROC=%6.4f RMSE=%6.4f | %s | %s [%1.0fs]\n", i, err, roc_avg, rmse_avg, estr, rstr, time1 - time0)
      else
        verbose && @printf("%3d: Acc=%6.4f ROC=%6.4f RMSE=%6.4f | %s | %s [%1.0fs]\n", i, err_avg, roc_avg, rmse_avg, estr, rstr, time1 - time0)
      end
    end
  end
  ## calculating prediction on training set
  train_cl = isempty(clamp) ?train_rat_all :makeClamped(train_rat_all, clamp)

  result = Dict{String,Any}()
  result["num_latent"]  = num_latent
  result["burnin"]      = burnin
  result["psamples"]    = psamples
  result["lambda_beta"] = data.entities[1].lambda_beta
  result["RMSE"]        = rmse_avg
  result["accuracy"]    = err_avg
  result["ROC"]         = roc_avg
  result["RMSE_train"]  = sqrt(mean( (getValues(data.relations[1].data) - train_cl) .^ 2 ))
  if full_prediction
    result["predictions_full"] = yhat_full / psamples
  end
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
