module BayesianDataFusion

using Distributions
using DataFrames

include("ROC.jl")
include("RelationData.jl")
include("macau.jl")

include("data_reading.jl")

end # module
