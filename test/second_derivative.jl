module TestSecondDerivative
    using Base.Test
    import FiniteDiff

    negative_sin(x) = -sin(x)
    negative_cos(x) = -cos(x)
    square(x) = x * x
    two(x) = 2.0

    function check_error(y_true, y_approximate, err = 1 // 10_000)
        # TODO: Decide how to handle very small inputs.
        if y_true > eps(y_true)
            @test abs(y_true - y_approximate) / abs(y_true) < err
        end
    end

    function run_tests(n::Integer)
        output = Array(Float64, 1)

        funcs = (
            (sin, negative_sin),
            (cos, negative_cos),
            (square, two),
            (exp, exp),
        )

        for (f, f′′) in funcs
            for _ in 1:n
                x = 1.0 + 100.0 * rand()

                y = FiniteDiff.second_derivative(f, x)
                check_error(f′′(x), y)

                FiniteDiff.second_derivative!(output, f, x)
                check_error(f′′(x), output[1])

                tmp = FiniteDiff.second_derivative(f, mutates=false)
                y = tmp(x)
                check_error(f′′(x), y)

                tmp! = FiniteDiff.second_derivative(f, mutates=true)
                tmp!(output, x)
                check_error(f′′(x), output[1])
            end
        end
    end

    @testset "second_derivative tests" begin
        run_tests(100)
    end
end
