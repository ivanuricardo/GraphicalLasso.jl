# GraphicalLasso.jl

[![Build Status](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ivanuricardo.github.io/GraphicalLasso.jl/stable)
[![codecov](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl/graph/badge.svg?token=f7OfqnmtEC)](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl)

This package provides efficient tools for generating sparse covariance matrices, estimating sparse precision matrices using the graphical lasso (glasso) algorithm (Friedman, Hastie, and Tibshirani 2008; Meinshausen and Bühlmann 2006), and selecting optimal regularization parameters.

## Key Features:

- **Sparse Covariance Matrix Generation**: Generate random sparse covariance matrices with customizable sparsity thresholds.
- **Graphical Lasso (glasso) Implementation**: Apply the glasso algorithm to estimate sparse precision matrices from empirical covariance matrices.
- **Extended Bayesian Information Criterion (EBIC)**: Calculate EBIC for model selection from Foygel and Drton (2010), supporting edge counting with thresholding.
- **Tuning Parameter Selection**: Automatically select optimal regularization parameters for the glasso algorithm using EBIC.
- **Covariance Matrix Validation**: Functions to check if a matrix is a valid covariance matrix (square, symmetric, positive semi-definite).

## Installation

To install this package, first enter into Pkg mode by pressing `]` in the Julia REPL, then run the following command:
```julia
pkg> add GraphicalLasso
```

There is also the option to install the development version of this package directly from the GitHub repository:
```julia
pkg> add https://github.com/ivanuricardo/GraphicalLasso.jl
```

## Functions Included:

- `softthresh(x, λ)`: Applies soft thresholding to an array.
- `cdlasso(W11, s12, λ; max_iter=100, tol=1e-5)`: Solves the coordinate descent Lasso problem.
- `glasso(s, obs, λ; penalizediag=true, γ=0.0, tol=1e-05, verbose=true, maxiter=100, winit=zeros(size(s)))`: Estimates a sparse inverse covariance matrix using the glasso algorithm.
- `countedges(x, thr)`: Counts the number of edges in an array exceeding a threshold.
- `ebic(θ, ll, obs, thr, γ)`: Calculates the EBIC for a given precision matrix.
- `critfunc(s, θ, rho; penalizediag=true)`: Computes the objective function for the graphical lasso.
- `tuningselect(s, obs, λ; γ=0.0)`: Selects the optimal regularization parameter using EBIC.
- `randsparsecov(p, thr)`: Generates a random sparse covariance matrix.
- `iscov(x)`: Checks if a matrix is a valid covariance matrix.

## Example

Here is an example of how to use this package to generate a sparse covariance matrix, apply the graphical lasso algorithm, and select the optimal tuning parameter:

```julia
using LinearAlgebra, GraphicalLasso, Random, Distributions
Random.seed!(123456)

# Generate true sparse covariance matrix
p = 20
thr = 0.5
Σ = randsparsecov(p, thr)

# Check if this is a valid covariance matrix
iscov(Σ)

# Generate data from the true covariance matrix, create sample covariance matrix
obs = 100
μ = zeros(p)
unstddf = rand(MvNormal(zeros(p), Σ), obs)
df = (unstddf .- mean(unstddf, dims=2)) ./ std(unstddf, dims=2)
s = df * df' / obs

# Select the optimal tuning parameter from a range
λvalues = 0.0:0.01:2.0
optimalλ = tuningselect(s, obs, λvalues, verbose=false)
println("Optimal λ: ", optimalλ)

# Apply the graphical lasso algorithm
result = glasso(s, obs, optimalλ)

# Extract results
W = result.W
θ = result.θ
ll = result.ll
bicval = result.bicval

println("Estimated Precision Matrix: ", θ)
println("Log-Likelihood: ", ll)
println("EBIC Value: ", bicval)

# Validate if the result is a valid covariance matrix
is_valid_cov = iscov(W)
println("Is the estimated matrix a valid covariance matrix? ", is_valid_cov)
```

Moreover, although not the main focus of this package, we also provide a method to compute the lasso solution via coordinate descent.
We demonstrate this method below with a generated data set and a sparse response vector.

```julia
using LinearAlgebra, GraphicalLasso, Random, Plots
Random.seed!(123456)
N = 100
k = 100
kzeros = 90
X = randn(N, k)
beta = ones(k)
beta[1:kzeros] .= 0
betahat = zeros(k)
y = X * beta + randn(N)

λ = 10.0
cdlassobeta = cdlasso(X'X, X'y, λ)

# We expect the last column to be dense.
heatmap(reshape(cdlassobeta, 10, 10), yflip = true)
```

## Contribution

We welcome contributions to improve the package.
If you encounter any issues or have suggestions for new features, feel free to open an issue or submit a pull request.

## References

- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3), 432-441.
- Foygel, R., & Drton, M. (2010). Extended Bayesian information criteria for Gaussian graphical models. In Advances in Neural Information Processing Systems (pp. 604-612).
- Meinshausen, N., & Bühlmann, P. (2006). High-dimensional graphs and variable selection with the lasso. The Annals of Statistics, 34(3), 1436-1462.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ivanuricardo/GraphicalLasso.jl/blob/main/LICENSE) file for details.
