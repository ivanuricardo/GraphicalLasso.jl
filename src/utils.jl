
offdiag(x) = x[findall(!iszero, ones(size(x)) - I)]

function randsparsecov(p, thr)
    s = randn(p, p)
    denseΣ = (s + s') / 2

    valsΣ = eigvals(denseΣ)
    vecsΣ = eigvecs(denseΣ)

    unthreshΣ = vecsΣ' * Diagonal(abs.(valsΣ)) * vecsΣ
    Σ = softthresh.(unthreshΣ, thr) + I

    return Hermitian(Σ)
end

function iscov(xx::AbstractMatrix{T}) where {T<:Real}
    n, m = size(xx)
    if n != m
        @info "result is not square."
        return false
    end

    if !issymmetric(xx)
        @info "result is not symmetric."
        return false
    end

    xxevals = eigvals(xx)
    if any(xxevals .< 0)
        @info "result is not positive semi-definite."
        return false
    end

    return true
end

