```@docs
    glasso(s::Matrix{Float64}, obs::Int, λ::Real; penalizediag::Bool=true, γ::Real=0.0, tol::Float64=1e-05, verbose::Bool=true, maxiter::Int=100, winit::Matrix{Float64}=zeros(size(s)))
    cdlasso(W11::Matrix{T}, s12::Vector{T}, λ::Real; max_iter::Int=100, tol::T=1e-5) where {T<:Real}
```
