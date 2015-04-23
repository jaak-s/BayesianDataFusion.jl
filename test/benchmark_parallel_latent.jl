using BayesianDataFusion

if nprocs() <= 1
  addprocs(8)
  @everywhere using BayesianDataFusion
end

W = sprand(1_500_000, 1000, 0.01)
rd = RelationData(W, class_cut = 0.5)
assignToTest!(rd.relations[1], 50)

############ num_latent = 10 ############
## single threaded
blas_set_num_threads(1)
r1 = macau(rd, burnin = 2, psamples = 2, num_latent=10);

blas_set_num_threads(8)
r1 = macau(rd, burnin = 2, psamples = 2, num_latent=10);

blas_set_num_threads(16)
r1 = macau(rd, burnin = 2, psamples = 2, num_latent=10);

## multi-threaded (2)
r = macau(rd, burnin = 2, psamples = 2, num_latent=10, latent_pids=[2,3], latent_blas_threads=1);
r = macau(rd, burnin = 2, psamples = 2, num_latent=10, latent_pids=[2,3], latent_blas_threads=2);

############ num_latent = 30 ############
blas_set_num_threads(1)
r1 = macau(rd, burnin = 2, psamples = 2, num_latent=30);

blas_set_num_threads(2)
r1 = macau(rd, burnin = 2, psamples = 2, num_latent=30);

blas_set_num_threads(4)
r1 = macau(rd, burnin = 2, psamples = 2, num_latent=30);

## multi-threaded (2)
r = macau(rd, burnin = 2, psamples = 2, num_latent=10, latent_pids=[2,3], latent_blas_threads=1);
r = macau(rd, burnin = 2, psamples = 2, num_latent=10, latent_pids=[2,3], latent_blas_threads=2);

## multi-threaded (4)
r = macau(rd, burnin = 2, psamples = 2, num_latent=10, latent_pids=[2,3,4,5], latent_blas_threads=1);

## multi-threaded (8)
r = macau(rd, burnin = 2, psamples = 2, num_latent=10, latent_pids=[2:9;], latent_blas_threads=1);

