module BayesianDataFusion

export IndexedDF, getData, getCount, removeSamples, getValues
export RelationData
export Relation, numData, numTest, assignToTest!, pred
export Entity
export BMRF
export load_mf1c

using Distributions
using DataFrames

include("ROC.jl")
include("RelationData.jl")
include("BMRF.jl")

end # module
