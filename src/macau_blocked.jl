export macau_blocked

function macau_blocked(data::RelationData;
              num_latent::Int = 10,
              burnin          = 500,
              psamples        = 200,
              verbose::Bool   = true)
  length(data.entities)  == 2 || error("macau_blocked only supports 2 entities (matrix).")
  length(data.relations) == 1 || error("macau_blocked only supports 1 relation, a matrix.")
  hasFeatures(data.entities[1]) || error("macau_blocked does not support features, entity '$(data.entities[1].name)' has features.")
  hasFeatures(data.entities[2]) || error("macau_blocked does not support features, entity '$(data.entities[2].name)' has features.")

  verbose && println("Model setup")
  reset!(data, num_latent)

  verbose && println("Sampling")
  err_avg  = 0.0
  roc_avg  = 0.0
  rmse_avg = 0.0

  local probe_rat_all::Vector{Float64}, clamped_rat_all::Vector{Float64}
  local probe_stdev::Vector{Float64}
  local train_rat_all

  blocks = Vector{Block}[
    make_blocks(data.entities[1]),
    make_blocks(data.entities[2])
  ]

  for i in 1 : burnin + psamples
    time0 = time()

    for j in 1:length(data.entities)
      mj = data.entities[j].model
      nu   = num_latent

      mj.mu, mj.Lambda = rand( ConditionalNormalWishart(mj.sample, mj.mu0, mj.b0, mj.WI, nu) )

      ## latent vectors
      sample_v = data.entities[j == 1 ? 2 : 1].model.sample
      sample_users_blocked_all!(blocks[j], mj.sample, sample_v, data.relations[1].model.alpha, mj.mu, mj.Lambda)
    end
    rel = data.relations[1]
    probe_rat = pred(rel, rel.test_vec, rel.test_F)

    time1 = time()
  end
end
