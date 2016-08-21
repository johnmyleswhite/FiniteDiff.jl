module TestGradient
    using Base.Test
    import FiniteDiff

    f1(x) = sin(x[1]) + cos(x[2])
    function g1!(out, x)
        out[1] = cos(x[1])
        out[2] = -sin(x[2])
        return
    end

    f2(x) = 2.3 * sin(x[1]) + 3.1 * cos(x[2])
    function g2!(out, x)
        out[1] = 2.3 * cos(x[1])
        out[2] = -3.1 * sin(x[2])
        return
    end

    f3(x) = 0.1 * (x[1] - 1.0)^2 - 0.3 * (x[2] - 0.7)^2
    function g3!(out, x)
        out[1] = 0.1 * 2.0 * (x[1] - 1.0)
        out[2] = -0.3 * 2.0 * (x[2] - 0.7)
        return
    end

    function check_error(y_true, y_approximate, err = 1 // 1_000)
        n = length(y_true)
        @test length(y_approximate) == n
        for i in 1:n
            if y_true[i] > eps(y_true[i])
                @test abs(y_true[i] - y_approximate[i]) / abs(y_true[i]) < err
            end
        end
    end

    function run_tests(n::Integer)
        funcs = (
            (f1, g1!),
            (f2, g2!),
            (f3, g3!),
        )

        x = Array(Float64, 2)
        output = Array(Float64, 2)
        buffer = Array(Float64, 2)
        buffer_complex = Array(Complex{Float64}, 2)
        gr = Array(Float64, 2)

        n_dims = length(x)

        modes = (
            FiniteDiff.ForwardMode(),
            FiniteDiff.BackwardMode(),
            FiniteDiff.CentralMode(),
            FiniteDiff.ComplexMode(),
        )

        for (f, f′!) in funcs
            for itr in 1:n
                for i in 1:n_dims
                    x[i] = itr
                end

                # Evaluate the true gradient and store it into gr.
                f′!(gr, x)

                for mode in modes
                    if mode != FiniteDiff.ComplexMode()
                        FiniteDiff.gradient!(output, f, x, buffer, mode)
                    else
                        FiniteDiff.gradient!(output, f, x, buffer_complex, mode)
                    end
                    check_error(gr, output)

                    output2 = FiniteDiff.gradient(f, x, mode)
                    check_error(gr, output2)

                    tmp! = FiniteDiff.gradient(f, mode, mutates = true)
                    if mode != FiniteDiff.ComplexMode()
                        tmp!(output, x, buffer)
                    else
                        tmp!(output, x, buffer_complex)
                    end
                    check_error(gr, output)

                    tmp = FiniteDiff.gradient(f, mode, mutates = false)
                    output2 = tmp(x)
                    check_error(gr, output2)
                end
            end
        end
    end

    @testset "gradient tests" begin
        run_tests(100)
    end
end
