module BayesianDataFusion

export IndexedDF, getData, removeSamples
export RelationData
export Relation, numData, numTest, assignToTest!
export Entity
export BMRF
export load_mf1c

using Distributions
using DataFrames

include("ROC.jl")
include("RelationData.jl")
include("BMRF.jl")

end # module
