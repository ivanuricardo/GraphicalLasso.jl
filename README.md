# GraphicalLasso.jl

[![Build Status](https://github.com/Ivan/GraphicalLasso.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Ivan/GraphicalLasso.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ivanuricardo.github.io/GraphicalLasso.jl/dev/)
[![codecov](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl/graph/badge.svg?token=f7OfqnmtEC)](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl)

A package for the Graphical Lasso in Julia.
We provide functions to solve the graphical lasso problem using coordinate descent and other related functionalities.
Much of the code is based on the implementation of the graphical lasso in the `glasso` package in R.
However, we provide extended BIC for the selection of the tuning parameter and functions to generate a random sparse covariance matrix.

## Example Usage

```julia
using GraphicalLasso

```
