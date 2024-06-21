
@testset "softthresh" begin
    @test softthresh(1.0, 0.5) ≈ 0.5
    @test softthresh(1.0, 1.0) ≈ 0.0

    v = [1.0, 2.0, -3.0]
    sv = softthresh(v, 1.5)
    @test sv[1] ≈ 0.0
    @test sv[2] ≈ 0.5
    @test sv[3] ≈ -1.5

end

@testset "off diagonal average" begin
    a = [2 1 1; 1 2 1; 1 1 2]
    @test mean(offdiag(a)) ≈ 1.0
end

@testset "counting edges" begin

    a = [1 2 3; 4 5 6; 7 8 9]
    a[diagind(a)] .= 0
    @test countedges(a, 1e-10) ≈ 6
end

@testset "Glasso returns covariance result" begin
    using Random
    Random.seed!(1234)

    nobs = 200
    df = randn(nobs, 5)
    s = cov(df)

    gs = glasso(s, nobs, 0.1)
    @test iscov(gs.W)
end

@testset "Random sparse matrix returns covariance" begin
    using Random
    Random.seed!(1234)

    s1 = randsparsecov(10, 0.5)
    s2 = randsparsecov(20, 0.2)
    s3 = randsparsecov(50, 0.5)
    s4 = randsparsecov(100, 0.5)

    @test iscov(s1)
    @test iscov(s2)
    @test iscov(s3)
    @test iscov(s4)
end

@testset "large tuning parameter" begin
    using Random
    Random.seed!(1234)

    nobs = 200
    df = randn(nobs, 10)
    s = cov(df)

    λ = 8e10

    gs = glasso(s, nobs, λ)
    od = offdiag(gs.θ)
    @test od ≈ zeros(90)
end

@testset "CD Lasso yields least squares solution" begin
    using Random
    Random.seed!(1234)

    nobs = 500
    X = randn(nobs, 10)

    trueβ = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = X * trueβ + 0.1 * randn(nobs)
    unmeanX = X .- mean(X, dims=1)
    stdX = unmeanX ./ std(X, dims=1)

    lsβ = inv(stdX' * stdX) * stdX' * y
    λ = 0.0
    cdlassoβ = cdlasso(stdX' * stdX, stdX' * y, λ; tol=1e-5)

    @test cdlassoβ ≈ lsβ
end

@testset "CD Lasso yields correct number of zeros" begin
    using Random
    Random.seed!(1234)

    nobs = 20
    X = randn(nobs, 10)

    trueβ = [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0]
    y = X * trueβ + 0.1 * randn(nobs)
    unmeanX = X .- mean(X, dims=1)
    stdX = unmeanX ./ std(X, dims=1)

    λ = 1.3
    cdlassoβ = cdlasso(stdX' * stdX, stdX' * y, λ; tol=1e-5)
    numedges = countedges(cdlassoβ, 1e-10)

    @test numedges ≈ 5
end

# @testset "Glasso tuning parameter" begin
#     using Random, Statistics, LinearAlgebra, Distributions
#     Random.seed!(1234)
#
#     p = 40
#     μ = zeros(p)
#     Σ = randsparsecov(p, 0.3)
#     nobs = 50
#
#     df = rand(MvNormal(μ, Σ), nobs)
#     stddf = df ./ std(df, dims=2)
#     s = cov(df')
#
#     λ = 0:0.01:1.2
#     tuning = tuningselect(s, nobs, λ; tol=1e-5, verbose=false)
#     gs = glasso(s, nobs, 1.2)
# end
