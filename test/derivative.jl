module TestDerivative
    using Base.Test
    using FiniteDiff

    srand(1)

    negsin(x) = -sin(x)
    negcos(x) = -cos(x)
    square(x) = x * x
    double(x) = x + x
    sqrt_deriv(x) = 1 / (2 * sqrt(x))

    function run_tests(n::Integer)
        out = Float64[0.0]

        funcs = (
            (sin, cos),
            (cos, negsin),
            (negsin, negcos),
            (square, double),
            (exp, exp),
            (log, inv),
            (sqrt, sqrt_deriv),
        )

        modes = (
            FiniteDiff.Forward,
            FiniteDiff.Backward,
            FiniteDiff.Central,
            FiniteDiff.Complex,
        )

        for (f, f′) in funcs
            for _ in 1:n
                x = 1.0 + rand()

                expected_errors = (
                    10 * FiniteDiff.@forward(x),
                    10 * FiniteDiff.@backward(x),
                    10 * FiniteDiff.@central(x)^2,
                    eps(0.0),
                )

                for i in 1:length(modes)
                    FiniteDiff.derivative!(out, f, x, modes[i])
                    @test abs(f′(x) - out[1]) < expected_errors[i]

                    y = FiniteDiff.derivative(f, x, modes[i])
                    @test abs(f′(x) - y) < expected_errors[i]

                    FiniteDiff.derivative(f, modes[i], mutates=true)(out, x)
                    @test abs(f′(x) - out[1]) < expected_errors[i]

                    y = FiniteDiff.derivative(f, modes[i], mutates=false)(x)
                    @test abs(f′(x) - y) < expected_errors[i]
                end
            end
        end
    end

    run_tests(10_000)
end
