module TestHessian
    using Base.Test
    import FiniteDiff

    negative_sin(x) = -sin(x)
    negative_cos(x) = -cos(x)

    f1(x) = sin(x[1]) + cos(x[2])
    function h1!(out, x)
        out[1, 1] = negative_sin(x[1])
        out[1, 2] = 0.0
        out[2, 1] = 0.0
        out[2, 2] = negative_cos(x[2])
        return
    end

    f2(x) = 2.3 * sin(x[1]) + 3.1 * cos(x[2])
    function h2!(out, x)
        out[1, 1] = 2.3 * negative_sin(x[1])
        out[1, 2] = 0.0
        out[2, 1] = 0.0
        out[2, 2] = 3.1 * negative_cos(x[2])
        return
    end

    f3(x) = 0.1 * (x[1] - 1.0)^2 - 0.3 * (x[2] - 0.7)^2
    function h3!(out, x)
        out[1, 1] = 0.1 * 2.0
        out[1, 2] = 0.0
        out[2, 1] = 0.0
        out[2, 2] = -0.3 * 2.0
        return
    end

    f4(x) = x[1]^2 * x[2]^2
    function h4!(out, x)
        out[1, 1] = 2.0 * x[2]^2
        out[1, 2] = 4.0 * x[1] * x[2]
        out[2, 1] = 4.0 * x[1] * x[2]
        out[2, 2] = 2.0 * x[1]^2
        return
    end

    function check_error(y_true, y_approximate, err = 1 // 10_000)
        n = length(y_true)
        @test length(y_approximate) == n
        for i in 1:n
            # TODO: Decide how to handle very small inputs.
            if y_true[i] > eps(y_true[i])
                @test abs(y_true[i] - y_approximate[i]) / abs(y_true[i]) < err
            end
        end
    end

    function run_tests(n::Integer)
        funcs = (
            (f1, h1!),
            (f2, h2!),
            (f3, h3!),
            (f4, h4!),
        )

        x = Array(Float64, 2)
        output = Array(Float64, 2, 2)
        buffer = Array(Float64, 2)
        H = Array(Float64, 2, 2)

        n_dims = length(x)

        for (f, f′′!) in funcs
            for _ in 1:n
                for i in 1:n_dims
                    x[i] = 100.0 * randn()
                end

                f′′!(H, x)

                FiniteDiff.hessian!(output, f, x, buffer)
                check_error(H, output)

                output2 = FiniteDiff.hessian(f, x)
                check_error(H, output2)

                tmp! = FiniteDiff.hessian(f, mutates = true)
                tmp!(output, x, buffer)
                check_error(H, output)

                tmp = FiniteDiff.hessian(f, mutates = false)
                output2 = tmp(x)
                check_error(H, output2)
            end
        end
    end

    @testset "hessian tests" begin
        run_tests(1)
    end
end
