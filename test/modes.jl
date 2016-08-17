module TestModes
    using Base.Test
    import FiniteDiff

    @testset "Mode type definitions and constructors" begin
        modes = (
            FiniteDiff.ForwardMode,
            FiniteDiff.BackwardMode,
            FiniteDiff.CentralMode,
            FiniteDiff.ComplexMode,
            FiniteDiff.HessianMode,
        )

        for mode in modes
            m = mode()
            @test isa(m, mode)
            @test isa(m, FiniteDiff.Mode)
        end
    end
end
