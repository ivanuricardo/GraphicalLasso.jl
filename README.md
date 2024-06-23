# GraphicalLasso.jl

[![Build Status](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ivanuricardo.github.io/GraphicalLasso.jl/stable)
[![codecov](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl/graph/badge.svg?token=f7OfqnmtEC)](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl)

This package provides efficient tools for generating sparse covariance matrices, estimating sparse precision matrices using the graphical lasso (glasso) algorithm, and selecting optimal regularization parameters.

## Key Features:

- **Sparse Covariance Matrix Generation**: Generate random sparse covariance matrices with customizable sparsity thresholds.
- **Graphical Lasso (glasso) Implementation**: Apply the glasso algorithm to estimate sparse precision matrices from empirical covariance matrices.
- **Extended Bayesian Information Criterion (EBIC)**: Calculate EBIC for model selection, supporting edge counting with thresholding.
- **Tuning Parameter Selection**: Automatically select optimal regularization parameters for the glasso algorithm using EBIC.
- **Covariance Matrix Validation**: Functions to check if a matrix is a valid covariance matrix (square, symmetric, positive semi-definite).

## Installation

To install this package, you can directly add the package GitHub to the Julia package manager:
```julia
using Pkg;
Pkg.add(url="https://github.com/ivanuricardo/GraphicalLasso.jl")
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
- `iscov(xx)`: Checks if a matrix is a valid covariance matrix.

## Example

Here is an example of how to use this package to generate a sparse covariance matrix, apply the graphical lasso algorithm, and select the optimal tuning parameter:

```julia
using LinearAlgebra

# Generate a random sparse covariance matrix
p = 5
thr = 0.1
sparse_cov = randsparsecov(p, thr)

# Number of observations
obs = 100

# Regularization parameter
λ = 0.2

# Apply the graphical lasso algorithm
result = glasso(sparse_cov, obs, λ)

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

# Select the optimal tuning parameter from a range
λ_values = 0.1:0.1:1.0
optimal_λ = tuningselect(sparse_cov, obs, λ_values)
println("Optimal λ: ", optimal_λ)
```

## Contribution

We welcome contributions to improve the package.
If you encounter any issues or have suggestions for new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ivanuricardo/GraphicalLasso.jl/blob/main/LICENSE) file for details.
