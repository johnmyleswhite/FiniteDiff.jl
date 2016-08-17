module TestStepSize
    using Base.Test
    import FiniteDiff

    @testset "Per-mode step size functions" begin
        modes = (
            FiniteDiff.ForwardMode,
            FiniteDiff.BackwardMode,
            FiniteDiff.CentralMode,
            FiniteDiff.ComplexMode,
            FiniteDiff.HessianMode,
        )

        for T in (Float32, Float64, BigFloat)
            for mode in modes
                ϵ0 = FiniteDiff.step_size(mode(), zero(T))
                @test isa(ϵ0, T)
                @test ϵ0 <= one(T)

                ϵ1 = FiniteDiff.step_size(mode(), one(T))
                @test isa(ϵ1, T)
                @test ϵ1 <= one(T)

                @test ϵ0 <= ϵ1
            end
        end
    end
end
