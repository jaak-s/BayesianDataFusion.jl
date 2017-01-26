# BayesianDataFusion.jl package

This gives reference and examples for [BayesianDataFusion.jl](https://github.com/jaak-s/BayesianDataFusion.jl).

## Features
`BayesianDataFusion.jl` provides parallel and highly optimized implementation for

*  Bayesian Probabilistic Matrix Factorization (BPMF)
*  Bayesian Probabilistic Tensor Factorization (BPTF)
*  Macau - Bayesian Multi-relational Factorization with Side Information

These methods allow to predict **unobserved values** in the matrices (or tensors). Since they are all Bayesian methods we can also measure the **uncertainty** of the predictions. BPMF and BPTF are special cases of Macau. Macau adds

*  use of **entity side information** to improve factorization (e.g, user and/or movie features for factorizing movie ratings)
*  use of **relation side information** to improve factorization  (e.g., data about when user went to see particular movie)
*  factorization of **several** matrices (and tensors) for an entity simultaneously.
*  Macau can handle high dimensional side-information, e.g., 1,000,000-dimensional user features.

## Installation
Inside Julia:
```julia
Pkg.clone("https://github.com/jaak-s/BayesianDataFusion.jl.git")
```

# Examples
Next we give simple examples of using **Macau** for movie ratings prediction from MovieLens data, which is included in the BayesianDataFusion package.

## MovieLens
We will use `macau` function to factorize (incompletely observed) matrix of movie ratings with **side information** for both users and movies. The side information contains basic features about users like age group and gender and genre information for movies. To run the example first install Julia library for reading matlab files
```julia
Pkg.add("MAT")
```
Example code
```julia
using BayesianDataFusion
using MAT
## load and setup data
pkgdir = Pkg.dir("BayesianDataFusion")
data   = matread("$pkgdir/data/movielens_1m.mat")

## setup entities, assigning side information through optional argument F
users  = Entity("users",  F=data["Fu"]);
movies = Entity("movies", F=data["Fv"]);

## setup the relation between users and movies, data from sparse matrix data["X"]
## first element in '[users, movies]' corresponds to rows and second to columns of data["X"]
ratings = Relation(data["X"], "ratings", [users, movies], class_cut = 2.5);

## assign 500,000 of the observed ratings randomly to the test set
assignToTest!(ratings, 500_000)

## precision of the ratings to 1.5 (i.e., variance of 1/1.5)
setPrecision!(ratings, 1.5)

## the model (with only one relation)
RD = RelationData(ratings)

## run Gibbs sampler of Macau with 10 latent dimensions, total of 100 burnin and 400 posterior samples
result = macau(RD, burnin=100, psamples=400, clamp=[1.0, 5.0], num_latent=10)
```
This model has only a single relation `ratings` between entities `users` and `movies`.
We use precision 1.5, which is known to be a good estimate of movie rating noise.
The optional parameter `clamp=[1.0, 5.0]` to `macau` thresholds the predictions to be between 1.0 and 5.0.
To build a model with larger latent dimension use, for example, `num_latent=30`.

Macau output shows the progress of the Gibbs sampler:
```
  1: Acc=0.836 ROC=0.500 RMSE=1.118 | use[U:  3.1 β:0.04 λ=21.] mov[U:  3.1 β:0.07 λ=10.] | rati[α=1.5] [4s]
  2: Acc=0.836 ROC=0.500 RMSE=1.118 | use[U:  4.4 β:0.03 λ=60.] mov[U:  4.4 β:0.04 λ=38.] | rati[α=1.5] [0s]
...
 80: Acc=0.864 ROC=0.829 RMSE=0.889 | use[U: 72.7 β:1.34 λ=4.5] mov[U:122.6 β:3.25 λ=3.1] | rati[α=1.5] [0s]
 81: Acc=0.864 ROC=0.829 RMSE=0.888 | use[U: 73.0 β:1.39 λ=5.3] mov[U:123.0 β:3.32 λ=3.2] | rati[α=1.5] [0s]
...
```
The Acc/ROC/RMSE are computed on the test ratings. Note the optional argument `class_cut = 2.5`, used for creating a `Relation`, defines the class boundary for computing accuracy (Acc) and AUC-ROC (ROC) values. 

An example result of the run is:
```
Dict{AbstractString,Any} with 10 entries:
  "latent_multi_threading" => true
  "psamples"               => 400
  "lambda_beta"            => 14.506891192240246
  "RMSE"                   => 0.8526296036293598
  "train_counts"           => 500000x2 DataFrames.DataFrame…
  "predictions"            => 500000x5 DataFrames.DataFrame…
  "burnin"                 => 100
  "num_latent"             => 10
  "accuracy"               => 0.870428
  "ROC"                    => 0.8485116969174835
```
where `result["predictions"]` gives predicted values and their standard deviation for the values in the test set. The `result` also contains ROC, RMSE and accuracy values for the test set.

## MovieLens w/o side-information
The above example used user and move features. You can easily factorize the ratings without them, which would correspond to classic **BPMF** method. Here is an example code
```julia
using BayesianDataFusion
using MAT
## load and setup data
pkgdir = Pkg.dir("BayesianDataFusion")
data   = matread("$pkgdir/data/movielens_1m.mat")

## setup entities, no features (F):
users  = Entity("users");
movies = Entity("movies");

## setup the relation between users and movies, data from sparse matrix data["X"]
## first element in '[users, movies]' corresponds to rows and second to columns of data["X"]
ratings = Relation(data["X"], "ratings", [users, movies], class_cut = 2.5);

## assign 500,000 of the observed ratings randomly to the test set
assignToTest!(ratings, 500_000)

## precision of the ratings to 1.5 (i.e., variance of 1/1.5)
setPrecision!(ratings, 1.5)

## the model (with only one relation)
RD = RelationData(ratings)

## run Gibbs sampler of Macau with 10 latent dimensions, total of 100 burnin and 400 posterior samples
result = macau(RD, burnin=100, psamples=400, clamp=[1.0, 5.0], num_latent=10)
```
In most applications the performance of pure BPMF is weaker compared to Macau. This is also true in the case of MovieLens dataset.

## Tensor factorization with side information
Here is an example of factorization of 3-tensor on a toy data of `compound x cell_line x gene` where two of the modes have side information. We first generate the toy dataset:
```julia
using BayesianDataFusion
using DataFrames

## generating artificial data
Ncompounds = 100
Ncell_lines = 10
Ngenes      = 50

A = randn(Ncompounds,  2);
B = randn(Ncell_lines, 2);
C = randn(Ngenes, 2);

X = Float64[ sum(A[i,:].*B[j,:].*C[k,:]) for i in 1:size(A,1), j in 1:size(B,1), k in 1:size(C,1)];

## adding artificial data into DataFrame
df = DataFrame(compound=Int64[], cell_line=Int64[], gene=Int64[], value=Float64[])
for i=1:size(A,1), j=1:size(B,1), k=1:size(C,1)
  push!(df, Any[i, j, k, X[i,j,k]])
end

## generating side information
Fcompounds = A * randn(2, 5);
Fgenes = C * randn(2, 10);
```
The dataframe object has an id for each mode and the last column gives the value of the tensor.
```
head(df)
6×4 DataFrames.DataFrame
│ Row │ compound │ cell_line │ gene │ value     │
├─────┼──────────┼───────────┼──────┼───────────┤
│ 1   │ 1        │ 1         │ 1    │ -0.112793 │
│ 2   │ 1        │ 1         │ 2    │ 0.555784  │
│ 3   │ 1        │ 1         │ 3    │ 0.116109  │
│ 4   │ 1        │ 1         │ 4    │ 0.106436  │
│ 5   │ 1        │ 1         │ 5    │ 0.661452  │
│ 6   │ 1        │ 1         │ 6    │ -0.483896 │
```

Next we set up the 3-tensor model and add side information to `compound` and `gene` data
```julia
## creating the three entities (compounds and genes have side information)
compound  = Entity("compound", F = Fcompounds);
cell_line = Entity("cell_line");
gene      = Entity("gene", F = Fgenes);

## creating Tensor relation
gene_expr = Relation(df, "GeneExpr", [compound, cell_line, gene], class_cut = 0.0)

## setting noise precision of the observations (1 / variance)
setPrecision!(gene_expr, 5.0)

## assign 5000 values to test set
assignToTest!(gene_expr, 5000)

## the model with one 3-tensor
RD = RelationData(gene_expr)

## perform factorization
result = macau(RD, burnin=100, psamples=900, num_latent=10)
```
The dataframe `result["predictions"]` gives the predictions on the test set.
For this toy dataset `num_latent=10` is sufficient, for larger datasets it should be increased.
The model we executed looks like this:
```java
julia> RD
[Relations]
  GeneExpr: compound--cell_line--gene, #known = 45000, #test = 5000, α = 5.00
[Entities]
  compound:    100 with 5 features (λ = sample)
 cell_line:     10 with no features
      gene:     50 with 10 features (λ = sample)
```

## Multi-relational models
Macau also provides factorization of models with multiple relations, where each entity can have also side information.
Here is an example where we have 3 entities: `users`, `movies`, `books` with 2 relations `movie_ratings` and `book_ratings`.
For illustration we add side information on `movies` (8-dimensional) and `books` (12-dimensional).
```julia
using BayesianDataFusion

## entities
Nusers  = 50
Nmovies = 25
Nbooks  = 20
users  = Entity("users")
movies = Entity("movies", F = randn(Nmovies, 8))  # side information
books  = Entity("books",  F = randn(Nbooks, 12))  # side information

## relations, using random data
movie_ratings = Relation(sprand(Nusers, Nmovies, 0.2), "mratings", [users, movies], class_cut = 0.5)
book_ratings  = Relation(sprand(Nusers, Nbooks, 0.3), "bratings", [users, books], class_cut = 0.5)

## assign 20 movie ratings to test set
assignToTest!(movie_ratings, 20)

## set precision of the observation noise (inverse of variance)
setPrecision!(movie_ratings, 1.5)
setPrecision!(book_ratings, 2.0)

## multi-relational model
RD = RelationData()
addRelation!(RD, movie_ratings)
addRelation!(RD, book_ratings)

## run Macau with 10 latent dimensions, total of 100 burnin and 400 posterior samples
result = macau(RD, burnin=100, psamples=400, num_latent=10)
```

Note that Macau reports RMSE on the test data of the first relation, which in this example was `movie_ratings`.
To see the model structure you can check the model (even before the run).
```
julia> RD
[Relations]
  mratings: users--movies, #known = 233, #test = 20, α = 1.50
  bratings: users--books, #known = 314, #test = 0, α = 2.00
[Entities]
     users:     50 with no features
    movies:     25 with 8 features (λ = sample)
     books:     20 with 12 features (λ = sample)
```

# Saving latent vectors
To save the sampled latent variables to disk macau has `output` parameter. If it is set to non-empty string `macau` saves all posterior latent variable matrices and uses the `output` value as a prefix for the file names. The prefix can also include the path, for example
```
result = macau(RD, output = "/home/user/mylatent")
```
which will save the files as `/home/user/mylatent-...`.
Every sampled latent matrix of every entity will be saved as a separate file, stored by default in CSV format. There is also an option to save the output in binary 32bit floats (which will reduce the space required by 2x). This can be enabled by using `output_type="binary"`.

To save the files using prefix `mylatent` into the current working directory use `output = "mylatent"`.

## Reading latent matrix files
The saved CSV files can be read by the standard `readdlm` function
```julia
U1 = readdlm("mylatent-entity1-01.csv", ',')
```
where "01" is the sample number that takes values from "01" (or "001") to `psamples`.

If the `output_type="binary"` was used, the written files can be read by using the function `read_binary_float32`, for example
```julia
using BayesianDataFusion
U1 = read_binary_float32("mylatent-entity1-01.binary")
```

## Saving link matrix (beta)
For the models that have features the sampled link matrices `beta` can be saved by setting both `output` to a prefix and `output_beta = true`. For example:
```
result = macau(RD, output = "/home/user/mylatent", output_beta = true)
```
The beta matrices can be read similarly to the latent matrix files as described above, using either `readdlm` for CSV files and `read_binary_float32` for binary files.

# Efficient storage of sparse matrices
The package also includes functions for writing and reading binary format for **sparse matrix**.
```julia
## random sparse matrix
X = sprand(100, 50, 0.1)
write_sparse_float32("X.sparse", X)

## reading back the rows, cols and values from the file
I, J, V = read_sparse_float32("X.sparse")
X2 = sparse(I, J, V)
```
