module TestDerivative
    using Base.Test
    import FiniteDiff

    negative_sin(x) = -sin(x)
    negative_cos(x) = -cos(x)
    square(x) = x * x
    double(x) = x + x
    sqrt_deriv(x) = 1 / (2 * sqrt(x))

    function check_error(y_true, y_approximate, err = 1 // 10_000)
        # TODO: Decide how to handle very small inputs.
        if y_true > eps(y_true)
            @test abs(y_true - y_approximate) / abs(y_true) < err
        end
    end

    function run_tests(n::Integer)
        output = Array(Float64, 1)

        funcs = (
            (sin, cos),
            (cos, negative_sin),
            (negative_sin, negative_cos),
            (square, double),
            (exp, exp),
            (log, inv),
            (sqrt, sqrt_deriv),
        )

        modes = (
            FiniteDiff.ForwardMode(),
            FiniteDiff.BackwardMode(),
            FiniteDiff.CentralMode(),
            FiniteDiff.ComplexMode(),
        )

        for (f, f′) in funcs
            for itr in 1:n
                x = 1.0 + itr

                for i in 1:length(modes)
                    m = modes[i]

                    y = FiniteDiff.derivative(f, x, m)
                    check_error(f′(x), y)

                    FiniteDiff.derivative!(output, f, x, m)
                    check_error(f′(x), output[1])

                    tmp = FiniteDiff.derivative(f, m, mutates=false)
                    y = tmp(x)
                    check_error(f′(x), y)

                    tmp! = FiniteDiff.derivative(f, m, mutates=true)
                    tmp!(output, x)
                    check_error(f′(x), output[1])
                end

                y = FiniteDiff.derivative(f, x)
                check_error(f′(x), y)
            end
        end
    end

    @testset "derivative tests" begin
        run_tests(100)
    end
end
