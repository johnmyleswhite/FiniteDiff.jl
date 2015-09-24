module TestSecondDerivative
    using Base.Test
    using FiniteDiff

    srand(1)

    negsin(x) = -sin(x)
    negcos(x) = -cos(x)
    square(x) = x * x
    two(x) = 2.0

    function run_tests(n::Integer)
        out = Float64[0.0]

        funcs = (
            (sin, negsin),
            (cos, negcos),
            (square, two),
            (exp, exp),
        )

        for (f, f′′) in funcs
            for _ in 1:n
                x = 1.0 + rand()

                FiniteDiff.second_derivative!(out, f, x)
                @test abs(f′′(x) - out[1]) < 10 * FiniteDiff.@hessian(x)^2

                y = FiniteDiff.second_derivative(f, x)
                @test abs(f′′(x) - y) < 10 * FiniteDiff.@hessian(x)^2

                FiniteDiff.second_derivative(f, mutates=true)(out, x)
                @test abs(f′′(x) - out[1]) < 10 * FiniteDiff.@hessian(x)^2

                y = FiniteDiff.second_derivative(f, mutates=false)(x)
                @test abs(f′′(x) - y) < 10 * FiniteDiff.@hessian(x)^2
            end
        end
    end

    run_tests(10_000)
end
