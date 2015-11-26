# BayesianDataFusion

[![Build Status](https://travis-ci.org/jaak-s/BayesianDataFusion.jl.svg?branch=master)](https://travis-ci.org/jaak-s/BayesianDataFusion.jl)

Implementation of data fusion methods in Julia, specifically **Macau**, **BPMF** (Bayesian Probabilistic Matrix Factorization). Supported features:
* Factorization of matrices (without or with side information)
* Factorization of tensors (without or with side information)
* Co-factorization of multiple matrices and tensors (any side information is
  possible)
* Side information inside relation
* Parallelization (multi-core and multi-node)

BayesianDataFusion uses Gibbs sampling to learn the latent vectors and link
matrices. In addition to predictions Gibbs sampling also provides estimates
of **standard deviation** and possible other metrics (that can be computed from
samples).

## Installation
```julia
Pkg.clone("https://github.com/jaak-s/BayesianDataFusion.jl.git")
```

## Examples and documentation
For examples, please see [documentation](http://jaak-s.github.io/BayesianDataFusion.jl/).
