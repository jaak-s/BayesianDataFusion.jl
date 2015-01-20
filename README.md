# BayesianDataFusion

[![Build Status](https://travis-ci.org/jaak-s/BayesianDataFusion.jl.svg?branch=master)](https://travis-ci.org/jaak-s/BayesianDataFusion.jl)

Implementation of data fusion methods.

## Example
```julia
using BayesianDataFusion
x = load_mf1c(normalize_feat = true)
result = BMRF(x)
```
