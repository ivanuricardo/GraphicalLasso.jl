
countedges(x, thr) = sum(abs.(x) .> thr)

"""
    ebic(θ, ll, obs, thr, γ)

Calculates the Extended Bayesian Information Criterion (EBIC) for a given precision matrix `θ`.
From the paper by Foygel and Drton (2010), the EBIC is defined as:

``\\text{EBIC} = -2 \\times \\text{Log-likelihood} + \\log(n) \\times \\mathbf{E} + 4 \\times \\gamma \\times \\mathbf{E} \\times \\log(p)``

where:
- n is the number of observations.
- p is the number of variables.
- gamma is a tuning parameter.
- The number of edges is calculated as the count of entries in theta that exceed a given threshold.

# Arguments
- `θ::AbstractMatrix`: Precision matrix.
- `ll::Real`: Log-likelihood.
- `obs::Int`: Number of observations.
- `thr::Real`: Threshold value for counting edges.
- `γ::Real`: EBIC tuning parameter.

# Returns
- `Real`: EBIC value.
"""
function ebic(θ, ll, obs, thr, γ)
    edgecount = countedges(θ, thr)
    ebicpen = 4 * γ * edgecount * log(size(θ, 1))
    return -2 * ll + (log(obs) * edgecount) + ebicpen
end

"""
    critfunc(s, θ, rho; penalizediag=true)

Calculates the objective function value for the graphical lasso.

# Arguments
- `s::AbstractMatrix`: Empirical covariance matrix.
- `θ::AbstractMatrix`: Precision matrix.
- `rho::Real`: Regularization parameter.
- `penalizediag::Bool=true`: Whether to penalize the diagonal entries. (optional)

# Returns
- `Real`: Value of the objective function.
"""
function critfunc(s, θ, rho; penalizediag=true)
    ψ = copy(θ)
    if !penalizediag
        ψ[diagind(ψ)] .= 0
    end
    crit = -logdet(θ) + tr(s * θ) + sum(abs.(rho * ψ))
    return crit
end

"""
    tuningselect(s::Matrix{Float64}, obs::Int, λ::AbstractVector{T}; γ::Real=0.0) where {T}

Selects the optimal regularization parameter `λ` for the graphical lasso using EBIC.

# Arguments
- `s::Matrix{Float64}`: Empirical covariance matrix.
- `obs::Int`: Number of observations.
- `λ::AbstractVector{T}`: Vector of regularization parameters to be tested.
- `γ::Real=0.0`: EBIC tuning parameter. (optional)

# Returns
- `T`: The optimal regularization parameter from the input vector `λ`.
"""
function tuningselect(
    s::AbstractMatrix{Float64},
    obs::Int,
    λ::AbstractVector{T};
    γ::Real=0.0,
    kwargs...
) where {T}
    sortedλ = sort(λ)
    p, _ = size(s)
    numλ = length(sortedλ)

    covarray = zeros(Float64, p, p, numλ)
    bicvec = Vector{Float64}(undef, numλ)

    W, _, _, bicval = glasso(s, obs, sortedλ[1]; γ, kwargs...)
    bicvec[1] = bicval
    covarray[:, :, 1] = W

    for i in 2:numλ
        nextW = covarray[:, :, i-1]
        glassoresult = glasso(s, obs, sortedλ[i]; γ, winit=nextW, kwargs...)
        covarray[:, :, i] = glassoresult.W
        bicvec[i] = glassoresult.bicval
    end

    lowestidx = argmin(bicvec)
    return sortedλ[lowestidx]
end
