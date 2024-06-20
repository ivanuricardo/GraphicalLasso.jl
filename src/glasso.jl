
softthresh(x, λ) = sign.(x) .* max.(abs.(x) .- λ, 0)
loglik(s, θ, rho) = logdet(θ) - tr(s * θ) - sum(abs.(s * rho))
edges(x, thr) = sum(abs.(x) .> thr)
bic(θ, ll, nobs, thr) = -2 * ll + log(nobs) * edges(θ, thr)
offdiag(x, p) = [x[i, j] for i in 1:p for j in 1:p if i != j]

# Function to perform coordinate descent for lasso regression
function cdlasso(
    W11::Matrix{T},
    s12::Vector{T},
    λ::Real;
    max_iter::Int=100,
    tol::T=1e-4) where {T<:Real}

    p = length(s12)
    β = zeros(p)

    for _ in 1:max_iter
        β_old = copy(β)

        for j in 1:p
            r_j = s12[j] - W11[1:end.!=j, j]' * β[1:end.!=j]
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
    nobs::Int,
    λ::Real;
    tol::Float64=0.001,
    maxiter::Int=100)
    p = size(s, 1)
    W = s + λ * I

    niter = 0
    for _ in 1:maxiter
        niter += 1
        W_old = copy(W)

        for j in 1:p
            # Partition W and s
            idx = setdiff(1:p, j)
            W11 = W[idx, idx]
            s12 = s[idx, j]

            # solve the lasso problem for β̂
            βhat = cdlasso(W11, s12, λ; max_iter=maxiter, tol=tol)

            # Update W
            W[idx, j] = W11 * βhat
            W[j, idx] = W[idx, j]'
        end

        # Check for convergence
        if mean(abs.(offdiag(W - W_old, p))) < tol
            @info "glasso converged with $niter iterations."
            break
        end
    end

    θ = inv(W)
    ll = loglik(s, θ, W)
    bicval = bic(θ, ll, nobs, tol)

    return (; W, θ, ll, bicval)
end

