
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

@testset "positive definite result" begin
    using Random
    Random.seed!(1234)

    nobs = 200
    df = randn(nobs, 5)
    s = cov(df)

    gs = glasso(s, nobs, 0.1)
    @test all(eigvals(gs.W) .> 0)
    @test all(eigvals(gs.θ) .> 0)
end

@testset "positive definite rand cov" begin
    using Random
    Random.seed!(1234)

    s = randsparsecov(10, 0.5)

    @test all(eigvals(s) .> 0)
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

@testset "Glasso tuning parameter" begin
    using Random
    Random.seed!(1234)
    nobs = 100
    df = randn(nobs, 10)
    Σ = cov(df)
    λ = 0:0.01:5
    tuning = tuningselect(s, nobs, λ; tol=1e-5)
    numedges = countedges(gs.θ, 1e-10)
    @test numedges ≈ 10
end
