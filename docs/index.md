# BayesianDataFusion.jl package

This gives reference and examples for [BayesianDataFusion.jl](https://github.com/jaak-s/BayesianDataFusion.jl).

## Features
`BayesianDataFusion.jl` provides parallel and highly optimized implementation for

*  Bayesian Probabilistic Matrix Factorization (BPMF)
*  Bayesian Probabilistic Tensor Factorization (BPTF)
*  Macau - Bayesian Multi-relational Factorization with Side Information

These methods allow to predict **unobserved values** in the matrices (or tensors). Since they are all Bayesian methods we can also measure the **uncertainty** of the predictions. BPMF and BPTF are special cases of Macau. Macau adds

*  use of entity *side information* to improve factorization (e.g, user and/or movie features for factorizing movie ratings)
*  use of relation *side information* to improve factorization  (e.g., data about when user went to see particular movie)
*  factorization of several matrices (and tensors) simultaneously.

## Installation
Inside Julia:
```julia
Pkg.clone("https://github.com/jaak-s/BayesianDataFusion.jl.git")
```

## Examples
Next we give simple examples of using **Macau** for movie ratings prediction from MovieLens data, which is included in the BayesianDataFusion package.

### MovieLens
Factorization of (incompletely observed) matrix of movie ratings with side information for both users and movies:
```julia
using BayesianDataFusion
using MAT
pkgdir = Pkg.dir("BayesianDataFusion")
data   = matread("$pkgdir/data/movielens_1m.mat")

## factorize ratings X with user features from Fu and movie features from Fv
RD = RelationData(data["X"], feat1=data["Fu"], feat2=data["Fv"])

## assign 500,000 of the observed values randomly to the test set
assignToTest!(RD.relations[1], 500_000)

## precision (inverse of variance) of the observations to 1.5.
setPrecision!(RD.relations[1], 1.5)

## view the model
RD

## run Gibbs sampler of Macau with 10 latent dimensions, total of 400 burnin and 200 posterior samples
result = macau(RD, burnin=400, psamples=200, clamp=[1.0, 5.0], num_latent=10)
```
This model has only a single relation, accessed by `RD.relations[1]`. We use precision 1.5, which is known to be a good estimate of movie rating noise. The optional parameter `clamp=[1.0, 5.0]` thresholds the predictions to be between 1.0 and 5.0.
