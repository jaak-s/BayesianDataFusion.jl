# BayesianDataFusion

[![Build Status](https://travis-ci.org/jaak-s/BayesianDataFusion.jl.svg?branch=master)](https://travis-ci.org/jaak-s/BayesianDataFusion.jl)

Implementation of data fusion methods.

## Installation
```julia
Pkg.clone("https://github.com/JuliaLang/IterativeSolvers.jl.git")
Pkg.clone("https://github.com/jaak-s/BayesianDataFusion.jl.git")
```

## Example
```julia
using BayesianDataFusion
x = load_mf1c(normalize_feat = true)
result = macau(x)
```
