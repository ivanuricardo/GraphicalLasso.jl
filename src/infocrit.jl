
countedges(x, thr) = sum(abs.(x) .> thr)

function ebic(θ, ll, obs, thr, γ)
    ebicpen = 4 * γ * countedges(θ, thr) * log(size(θ, 1))
    return -2 * ll + (log(obs) * countedges(θ, thr)) + ebicpen
end

function critfunc(s, θ, rho; penalizediag=true)
    ψ = copy(θ)
    if !penalizediag
        ψ[diagind(ψ)] .= 0
    end
    crit = -logdet(θ) + tr(s * θ) + sum(abs.(rho * ψ))
    return crit
end

function tuningselect(
    s::Matrix{Float64},
    obs::Int,
    λ::AbstractVector{T};
    γ::Real=0.0
) where {T}
    sortedλ = sort(λ)
    p, _ = size(s)
    num_λ = length(sortedλ)

    covarray = Array{Float64,3}(undef, p, p, num_λ)
    bicvec = Vector{Float64}(undef, num_λ)

    W, _, _, bicval = glasso(s, obs, sortedλ[1]; γ)
    bicvec[1] = bicval
    covarray[:, :, 1] = W

    for i in 2:num_λ
        nextW = covarray[:, :, i-1]
        glassoresult = glasso(s, obs, sortedλ[i]; γ, winit=nextW)
        covarray[:, :, i] = glassoresult.W
        bicvec[i] = glassoresult.bicval
    end

    lowestidx = argmin(bicvec)
    return sortedλ[lowestidx]
end
