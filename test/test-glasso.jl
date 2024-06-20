
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
    @test mean(offdiag(a, 3)) ≈ 1.0
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
    df = randn(nobs, 10)
    s = df' * df / nobs

    gs = glasso(s, nobs, 0.1)
    @test isposdef(gs.W)
end
