# GraphicalLasso.jl

A package for fitting the graphical lasso and some diagnostics for tuning parameter selection.
This package follows the work of Friedman et al. (2008) and the extended BIC criterion of Foygel and Drton (2010).
We gain inspiration from the `glasso` package in R, and aim to provide a similar user experience in Julia.

## Graphical Lasso Main Functions

```@docs
glasso
cdlasso
```

## Information Criteria
```@docs
ebic
critfunc
tuningselect
```

## Utility Functions

```@docs
randsparsecov
iscov
```

## Index

```@index
```
