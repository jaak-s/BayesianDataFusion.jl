# BayesianDataFusion.jl package

This gives reference and examples for [BayesianDataFusion.jl](https://github.com/jaak-s/BayesianDataFusion.jl).

## Features
`BayesianDataFusion.jl` provides parallel and highly optimized implementation for

*  Bayesian Probabilistic Matrix Factorization (BPMF)
*  Bayesian Probabilistic Tensor Factorization (BPTF)
*  Macau - Bayesian Multi-relational Factorization with Side Information

BPMF and BPTF are special cases of Macau. Macau adds

*  *side information* for entities (e.g, user and/or movie features for factorizing movie ratings)
*  *side information* for relations (e.g., when the user went to see particular movie)
*  factorization of several matrices (and tensors) simultaneously.

These methods allow to predict **unobserved values** in the matrices (or tensors). Since they are all Bayesian method we can also measure the uncertainty.

## Installation
Inside Julia do following:
```julia
Pkg.clone("https://github.com/jaak-s/BayesianDataFusion.jl.git")
```

## Usage examples
Factorization of (incompletely observed) matrix of movie ratings with side information for both users and movies:
```julia
using BayesianDataFusion
using MAT
```
