
offdiag(x) = x[findall(!iszero, ones(size(x)) - I)]

"""
    randsparsecov(p, thr)

Generates a random sparse covariance matrix of size `p x p` with a specified threshold for sparsity.

# Arguments
- `p::Int`: The dimension of the covariance matrix.
- `thr::Real`: Threshold value for sparsity. Values below this threshold will be set to zero.

# Returns
- `Hermitian{Float64}`: A sparse covariance matrix.
"""
function randsparsecov(p, thr)
    s = randn(p, p)
    denseΣ = (s + s') / 2

    valsΣ = eigvals(denseΣ)
    vecsΣ = eigvecs(denseΣ)

    unthreshΣ = vecsΣ' * Diagonal(abs.(valsΣ)) * vecsΣ
    Σ = softthresh.(unthreshΣ, thr) + I

    return Hermitian(Σ)
end

"""
    iscov(xx::AbstractMatrix{T}) where {T<:Real}

Checks if a given matrix is a valid covariance matrix. A valid covariance matrix must be square, symmetric, and positive semi-definite.

# Arguments
- `xx::AbstractMatrix{T}`: Input matrix to check.

# Returns
- `Bool`: `true` if the matrix is a valid covariance matrix, `false` otherwise.
"""
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

