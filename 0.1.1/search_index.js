var documenterSearchIndex = {"docs":
[{"location":"#GraphicalLasso.jl","page":"GraphicalLasso.jl","title":"GraphicalLasso.jl","text":"","category":"section"},{"location":"","page":"GraphicalLasso.jl","title":"GraphicalLasso.jl","text":"A package for fitting the graphical lasso and some diagnostics for tuning parameter selection. This package follows the work of Friedman et al. (2008) and the extended BIC criterion of Foygel and Drton (2010). We gain inspiration from the glasso package in R, and aim to provide a similar user experience in Julia.","category":"page"},{"location":"#Graphical-Lasso-Main-Functions","page":"GraphicalLasso.jl","title":"Graphical Lasso Main Functions","text":"","category":"section"},{"location":"","page":"GraphicalLasso.jl","title":"GraphicalLasso.jl","text":"glasso\ncdlasso","category":"page"},{"location":"#GraphicalLasso.glasso","page":"GraphicalLasso.jl","title":"GraphicalLasso.glasso","text":"glasso(s::Matrix{Float64}, obs::Int, λ::Real; penalizediag::Bool=true, γ::Real=0.0, tol::Float64=1e-05, verbose::Bool=true, maxiter::Int=100, winit::Matrix{Float64}=zeros(size(s)))\n\nApplies the graphical lasso (glasso) algorithm to estimate a sparse inverse covariance matrix.\n\nArguments\n\ns::Matrix{Float64}: Empirical covariance matrix.\nobs::Int: Number of observations.\nλ::Real: Regularization parameter for the lasso penalty.\npenalizediag::Bool=true: Whether to penalize the diagonal entries. (optional)\nγ::Real=0.0: EBIC tuning parameter. (optional)\ntol::Float64=1e-05: Tolerance for the convergence criteria. (optional)\nverbose::Bool=true: If true, prints convergence information. (optional)\nmaxiter::Int=100: Maximum number of iterations. (optional)\nwinit::Matrix{Float64}=zeros(size(s)): Initial value of the precision matrix. (optional)\n\nReturns\n\nNamedTuple: A named tuple with fields:\nW::Matrix{Float64}: Estimated precision matrix.\nθ::Matrix{Float64}: Estimated inverse covariance matrix.\nll::Float64: Log-likelihood of the estimate.\nbicval::Float64: EBIC value of the estimate.\n\n\n\n\n\n","category":"function"},{"location":"#GraphicalLasso.cdlasso","page":"GraphicalLasso.jl","title":"GraphicalLasso.cdlasso","text":"cdlasso(W11::Matrix{T}, s12::Vector{T}, λ::Real; max_iter::Int=100, tol::T=1e-5) where {T<:Real}\n\nSolves the coordinate descent Lasso problem.\n\nArguments\n\nW11::Matrix{T}: A square matrix used in the coordinate descent update.\ns12::Vector{T}: A vector used in the coordinate descent update.\nλ::Real: Regularization parameter.\nmax_iter::Int=100: Maximum number of iterations. (optional)\ntol::T=1e-5: Tolerance for the convergence criteria. (optional)\n\nReturns\n\nVector{T}: Solution vector β.\n\n\n\n\n\n","category":"function"},{"location":"#Information-Criteria","page":"GraphicalLasso.jl","title":"Information Criteria","text":"","category":"section"},{"location":"","page":"GraphicalLasso.jl","title":"GraphicalLasso.jl","text":"ebic\ncritfunc\ntuningselect","category":"page"},{"location":"#GraphicalLasso.ebic","page":"GraphicalLasso.jl","title":"GraphicalLasso.ebic","text":"ebic(θ, ll, obs, thr, γ)\n\nCalculates the Extended Bayesian Information Criterion (EBIC) for a given precision matrix θ. From the paper by Foygel and Drton (2010), the EBIC is defined as:\n\ntextEBIC = -2 times textLog-likelihood + log(n) times mathbfE + 4 times gamma times mathbfE times log(p)\n\nwhere:\n\nn is the number of observations.\np is the number of variables.\ngamma is a tuning parameter.\nThe number of edges is calculated as the count of entries in theta that exceed a given threshold.\n\nArguments\n\nθ::AbstractMatrix: Precision matrix.\nll::Real: Log-likelihood.\nobs::Int: Number of observations.\nthr::Real: Threshold value for counting edges.\nγ::Real: EBIC tuning parameter.\n\nReturns\n\nReal: EBIC value.\n\n\n\n\n\n","category":"function"},{"location":"#GraphicalLasso.critfunc","page":"GraphicalLasso.jl","title":"GraphicalLasso.critfunc","text":"critfunc(s, θ, rho; penalizediag=true)\n\nCalculates the objective function value for the graphical lasso.\n\nArguments\n\ns::AbstractMatrix: Empirical covariance matrix.\nθ::AbstractMatrix: Precision matrix.\nrho::Real: Regularization parameter.\npenalizediag::Bool=true: Whether to penalize the diagonal entries. (optional)\n\nReturns\n\nReal: Value of the objective function.\n\n\n\n\n\n","category":"function"},{"location":"#GraphicalLasso.tuningselect","page":"GraphicalLasso.jl","title":"GraphicalLasso.tuningselect","text":"tuningselect(s::Matrix{Float64}, obs::Int, λ::AbstractVector{T}; γ::Real=0.0) where {T}\n\nSelects the optimal regularization parameter λ for the graphical lasso using EBIC.\n\nArguments\n\ns::Matrix{Float64}: Empirical covariance matrix.\nobs::Int: Number of observations.\nλ::AbstractVector{T}: Vector of regularization parameters to be tested.\nγ::Real=0.0: EBIC tuning parameter. (optional)\n\nReturns\n\nT: The optimal regularization parameter from the input vector λ.\n\n\n\n\n\n","category":"function"},{"location":"#Utility-Functions","page":"GraphicalLasso.jl","title":"Utility Functions","text":"","category":"section"},{"location":"","page":"GraphicalLasso.jl","title":"GraphicalLasso.jl","text":"randsparsecov\niscov","category":"page"},{"location":"#GraphicalLasso.randsparsecov","page":"GraphicalLasso.jl","title":"GraphicalLasso.randsparsecov","text":"randsparsecov(p, thr)\n\nGenerates a random sparse covariance matrix of size p x p with a specified threshold for sparsity.\n\nArguments\n\np::Int: The dimension of the covariance matrix.\nthr::Real: Threshold value for sparsity. Values below this threshold will be set to zero.\n\nReturns\n\nHermitian{Float64}: A sparse covariance matrix.\n\n\n\n\n\n","category":"function"},{"location":"#GraphicalLasso.iscov","page":"GraphicalLasso.jl","title":"GraphicalLasso.iscov","text":"iscov(xx::AbstractMatrix{T}) where {T<:Real}\n\nChecks if a given matrix is a valid covariance matrix. A valid covariance matrix must be square, symmetric, and positive semi-definite.\n\nArguments\n\nxx::AbstractMatrix{T}: Input matrix to check.\n\nReturns\n\nBool: true if the matrix is a valid covariance matrix, false otherwise.\n\n\n\n\n\n","category":"function"},{"location":"#Index","page":"GraphicalLasso.jl","title":"Index","text":"","category":"section"},{"location":"","page":"GraphicalLasso.jl","title":"GraphicalLasso.jl","text":"","category":"page"}]
}
