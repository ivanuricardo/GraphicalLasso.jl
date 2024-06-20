
@testset "softthresh" begin
    @test softthresh(1.0, 0.5) ≈ 0.5
    @test softthresh(1.0, 1.0) ≈ 0.0

    v = [1.0, 2.0, -3.0]
    sv = softthresh(v, 1.5)
    @test sv[1] ≈ 0.0
    @test sv[2] ≈ 0.5
    @test sv[3] ≈ -1.5

end
