
softthresh(x, λ) = sign.(x) .* max.(abs.(x) .- λ, 0)

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

        if norm(β - β_old) < tol
            break
        end
    end

    return β
end

function glasso(
    s::Matrix{Float64},
    obs::Int,
    λ::Real;
    penalizediag::Bool=true,
    γ::Real=0.0,
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
            idx = setdiff(1:p, j)
            W11 = W[idx, idx]
            s12 = s[idx, j]

            βhat = cdlasso(W11, s12, λ; max_iter=maxiter, tol=tol)

            W[idx, j] = W11 * βhat
            W[j, idx] = W[idx, j]'
        end

        if norm(W - W_old, 1) < tol
            if verbose
                @info "glasso converged with $niter iterations."
            end
            break
        end
    end

    θ = inv(W)
    ll = -(obs / 2) * critfunc(s, θ, W; penalizediag)
    bicval = ebic(θ, ll, obs, 1e-06, γ)

    return (; W, θ, ll, bicval)
end

