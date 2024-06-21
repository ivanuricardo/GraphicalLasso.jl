
softthresh(x, λ) = sign.(x) .* max.(abs.(x) .- λ, 0)
countedges(x, thr) = sum(abs.(x) .> thr)
bic(θ, ll, obs, thr) = -2 * ll + (log(obs) * countedges(θ, thr)) / obs
offdiag(x) = x[findall(!iszero, ones(size(x)) - I)]

function critfunc(s, θ, rho; penalizediag=true)
    ψ = copy(θ)
    if !penalizediag
        ψ[diagind(ψ)] .= 0
    end
    crit = -logdet(θ) + tr(s * θ) + sum(abs.(rho * ψ))
    return crit
end

# Function to perform coordinate descent for lasso regression
function cdlasso(
    W11::Matrix{T},
    s12::Vector{T},
    λ::Real;
    max_iter::Int=100,
    tol::T=1e-5) where {T<:Real}

    p = length(s12)
    β = zeros(p)

    for _ in 1:max_iter
        β_old = copy(β)

        for j in 1:p
            idx = setdiff(1:p, j)
            r_j = s12[j] - W11[idx, j]' * β[idx]
            β[j] = softthresh(r_j, λ) / W11[j, j]
        end

        # Check for convergence
        if norm(β - β_old) < tol
            break
        end
    end

    return β
end

# Function to perform graphical lasso
function glasso(
    s::Matrix{Float64},
    obs::Int,
    λ::Real;
    penalizediag::Bool=true,
    tol::Float64=1e-05,
    verbose::Bool=true,
    maxiter::Int=100,
    winit::Matrix{Float64}=zeros(size(s)),
)

    p = size(s, 1)
    if winit == zeros(size(s))
        W = copy(s) + (penalizediag ? λ * I : zero(s))
    else
        W = copy(winit)
    end

    niter = 0
    for _ in 1:maxiter
        niter += 1
        W_old = copy(W)

        for j in 1:p
            # Partition W and s
            idx = setdiff(1:p, j)
            W11 = W[idx, idx]
            s12 = s[idx, j]

            # solve the lasso problem for βhat
            βhat = cdlasso(W11, s12, λ; max_iter=maxiter, tol=tol)

            # Update W
            W[idx, j] = W11 * βhat
            W[j, idx] = W[idx, j]'
        end

        if mean(abs.(offdiag(W - W_old))) < tol
            if verbose
                @info "glasso converged with $niter iterations."
            end
            break
        end
    end

    θ = inv(W)
    ll = -(obs / 2) * critfunc(s, θ, W; penalizediag)
    bicval = bic(θ, ll, obs, 1e-06)

    return (; W, θ, ll, bicval)
end

function tuningselect(
    s::Matrix{Float64},
    obs::Int,
    λ::AbstractVector;
    kwargs...
)
    sortedλ = sort(λ)
    W, θ, ll, bicval = glasso(s, obs, sortedλ[1]; kwargs...)
    bicvec = fill(NaN, length(λ))
    bicvec[1] = bicval

    for i in 2:length(λ)
        W, θ, ll, bicval = glasso(s, obs, sortedλ[i]; kwargs...)
        bicvec[i] = bicval
    end

    lowestidx = argmin(bicvec)
    return sortedλ[lowestidx]

end

function randsparsecov(p, thr)
    s = randn(p, p)
    denseΣ = (s + s') / 2

    preΣ = softthresh.(denseΣ, thr)
    valsΣ = eigvals(preΣ)
    vecsΣ = eigvecs(preΣ)

    Σ = vecsΣ' * Diagonal(abs.(valsΣ)) * vecsΣ

    return Σ
end

