
countedges(x, thr) = sum(abs.(x) .> thr)

function ebic(θ, ll, obs, thr, γ)
    edgecount = countedges(θ, thr)
    ebicpen = 4 * γ * edgecount * log(size(θ, 1))
    return -2 * ll + (log(obs) * edgecount) + ebicpen
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
    numλ = length(sortedλ)

    covarray = zeros(Float64, p, p, numλ)
    bicvec = Vector{Float64}(undef, numλ)

    W, _, _, bicval = glasso(s, obs, sortedλ[1]; γ)
    bicvec[1] = bicval
    covarray[:, :, 1] = W

    for i in 2:numλ
        nextW = covarray[:, :, i-1]
        glassoresult = glasso(s, obs, sortedλ[i]; γ, winit=nextW)
        covarray[:, :, i] = glassoresult.W
        bicvec[i] = glassoresult.bicval
    end

    lowestidx = argmin(bicvec)
    return sortedλ[lowestidx]
end
