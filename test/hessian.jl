module TestHessian
    using Base.Test
    using FiniteDiff

    srand(1)

    negsin(x) = -sin(x)
    negcos(x) = -cos(x)

    f1(x) = sin(x[1]) + cos(x[2])
    function h1!(out, x)
        out[1, 1] = negsin(x[1])
        out[1, 2] = 0.0
        out[2, 1] = 0.0
        out[2, 2] = negcos(x[2])
        nothing
    end

    f2(x) = 2.3 * sin(x[1]) + 3.1 * cos(x[2])
    function h2!(out, x)
        out[1, 1] = 2.3 * negsin(x[1])
        out[1, 2] = 0.0
        out[2, 1] = 0.0
        out[2, 2] = 3.1 * negcos(x[2])
        nothing
    end

    f3(x) = 0.1 * (x[1] - 1.0)^2 - 0.3 * (x[2] - 0.7)^2
    function h3!(out, x)
        out[1, 1] = 0.1 * 2.0
        out[1, 2] = 0.0
        out[2, 1] = 0.0
        out[2, 2] = -0.3 * 2.0
        nothing
    end

    f4(x) = x[1]^2 * x[2]^2
    function h4!(out, x)
        out[1, 1] = 2.0 * x[2]^2
        out[1, 2] = 4.0 * x[1] * x[2]
        out[2, 1] = 4.0 * x[1] * x[2]
        out[2, 2] = 2.0 * x[1]^2
        nothing
    end

    function run_tests(n::Integer)
        funcs = (
            (f1, h1!),
            (f2, h2!),
            (f3, h3!),
            (f4, h4!),
        )

        x = Array(Float64, 2)
        out = Array(Float64, 2, 2)
        H = Array(Float64, 2, 2)

        n_dims = length(x)

        for (f, f′′!) in funcs
            for _ in 1:n
                for i in 1:n_dims
                    x[i] = randn()
                end

                f′′!(H, x)

                FiniteDiff.hessian!(out, f, x)
                for d1 in 1:n_dims
                    for d2 in 1:n_dims
                        @test abs(out[d1, d2] - H[d1, d2]) < 10 * max(
                            FiniteDiff.@hessian(x[d1]),
                            FiniteDiff.@hessian(x[d2]),
                        )
                    end
                end

                out2 = FiniteDiff.hessian(f, x)
                for d1 in 1:n_dims
                    for d2 in 1:n_dims
                        @test abs(out2[d1, d2] - H[d1, d2]) < 10 * max(
                            FiniteDiff.@hessian(x[d1]),
                            FiniteDiff.@hessian(x[d2]),
                        )
                    end
                end

                FiniteDiff.hessian(f, mutates = true)(out, x)
                for d1 in 1:n_dims
                    for d2 in 1:n_dims
                        @test abs(out2[d1, d2] - H[d1, d2]) < 10 * max(
                            FiniteDiff.@hessian(x[d1]),
                            FiniteDiff.@hessian(x[d2]),
                        )
                    end
                end

                out2 = FiniteDiff.hessian(f, mutates = false)(x)
                for d1 in 1:n_dims
                    for d2 in 1:n_dims
                        @test abs(out2[d1, d2] - H[d1, d2]) < 10 * max(
                            FiniteDiff.@hessian(x[d1]),
                            FiniteDiff.@hessian(x[d2]),
                        )
                    end
                end
            end
        end
    end

    run_tests(10_000)
end
