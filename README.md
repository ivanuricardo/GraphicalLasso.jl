# GraphicalLasso.jl

[![Build Status](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ivanuricardo/GraphicalLasso.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![][docs-stable-img]][docs-stable-url] [![build](https://github.com/ivanuricardo/GraphicalLasso.jl/workflows/CI/badge.svg)](https://github.com/ivanuricardo/GraphicalLasso.jl/actions?query=workflow%3ACI) 
[![codecov](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl/graph/badge.svg?token=f7OfqnmtEC)](https://codecov.io/gh/ivanuricardo/GraphicalLasso.jl)
[docs-stable-img]: https://img.shields.io/badge/docs-blue.svg
[docs-stable-url]: https://ivanuricardo.github.io/GraphicalLasso.jl/latest

A package for the Graphical Lasso in Julia.
We provide functions to solve the graphical lasso problem using coordinate descent and other related functionalities.
Much of the code is based on the implementation of the graphical lasso in the `glasso` package in R.
However, we provide extended BIC for the selection of the tuning parameter and functions to generate a random sparse covariance matrix.

## Example Usage

```julia
using GraphicalLasso

```
