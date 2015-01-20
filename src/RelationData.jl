using DataFrames

type RelationData{XT,YT}
  F::XT
  Ft::XT
  Am::YT
  Au::YT
  num_p::Int
  num_m::Int
  probe_vec
  ratings_test
  mean_rating::Float64
  RelationData(F, Am) = new(F, F', Am, Am', size(Am,1), size(Am,2))
end

function normalizeFeatures!(data::RelationData)
  diagsq  = sqrt(vec( sum(data.F .^ 2,1) ))
  data.F  = data.F * spdiagm(1.0 ./ diagsq)
  data.Ft = data.F'
  return
end

function load_mf1c(ic50_file     = "chembl_19_mf1c/chembl-IC50-346targets.csv",
                   cmp_feat_file = "chembl_19_mf1c/chembl-IC50-compound-feat.csv")
  ## reading IC50 matrix
  X = readtable(ic50_file, header=true)
  rename!(X, [:row, :col], [:compound, :target])

  X[:, :value] = log10(X[:, :value]) + 1e-5
  idx          = sample(1:size(X,1), int(floor(20/100 * size(X,1))); replace=false)
  probe_vec    = array(X[idx,:])
  X            = X[setdiff(1:size(X,1), idx), :]
  
  ## reading feature matrix
  feat = readtable(cmp_feat_file, header=true)
  F    = sparse(feat[:compound], feat[:feature], 1.0)

  Am = sparse( X[:compound], X[:target], X[:value])
  num_p, num_m = size(Am)
  
  ## creating data object
  data = RelationData{typeof(F), typeof(Am)}(F, Am)
  data.probe_vec    = probe_vec
  data.ratings_test = data.probe_vec[:,3] .< log10(200)
  data.mean_rating  = sum(Am) / size(X,1)
  
  return data
end
