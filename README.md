# GraphicalLasso.jl

[![Build Status](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ivanuricardo.github.io/GraphicalLasso.jl/stable)
[![codecov](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl/graph/badge.svg?token=f7OfqnmtEC)](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl)

This package provides efficient tools for generating sparse covariance matrices, estimating sparse precision matrices using the graphical lasso (glasso) algorithm, and selecting optimal regularization parameters.

## Key Features:

- **Sparse Covariance Matrix Generation**: Generate random sparse covariance matrices with customizable sparsity thresholds.
- **Graphical Lasso Implementation**: Apply the glasso algorithm to estimate sparse precision matrices from empirical covariance matrices.
- **Extended Bayesian Information Criterion (EBIC)**: Calculate EBIC for model selection, supporting edge counting with thresholding.
- **Tuning Parameter Selection**: Automatically select optimal regularization parameters for the glasso algorithm using EBIC.
- **Covariance Matrix Validation**: Functions to check if a matrix is a valid covariance matrix (square, symmetric, positive semi-definite).

## Installation

To install this package, you can directly add the package github to the Julia package manager:
```julia
using Pkg;
```

