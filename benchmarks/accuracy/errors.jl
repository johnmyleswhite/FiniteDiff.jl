import FiniteDiff
import Calculus
import ForwardDiff

# TODO: derivative, second_derivative

include("../test_functions.jl")
include("../gradient_functions.jl")
include("../hessian_functions.jl")
include("../max_error.jl")

io = open("errors.tsv", "w")

@printf(io, "timestamp\tf\tx\tmethod\terr\tlatency\n")

for test_func in test_funcs
    for x1 in test_func.x1s
        for x2 in test_func.x2s
            x = [x1, x2]

            g_true = ForwardDiff.gradient(test_func.f, x)
            h_true = ForwardDiff.hessian(test_func.f, x)

            # TODO: Make these latencies less crazy for the first run of each
            # function.
            for grad_func in grad_funcs
                latency = @elapsed g_approx = grad_func.f(test_func.f, x)
                @printf(
                    io,
                    "%s\t%s\t%s\t%s\t%s\t%s\n",
                    Dates.now(),
                    test_func.name,
                    x,
                    grad_func.name,
                    max_error(g_true, g_approx),
                    latency,
                )
            end

            for hess_func in hess_funcs
                latency = @elapsed h_approx = hess_func.f(test_func.f, x)
                @printf(
                    io,
                    "%s\t%s\t%s\t%s\t%s\t%s\n",
                    Dates.now(),
                    test_func.name,
                    x,
                    hess_func.name,
                    max_error(h_true, h_approx),
                    latency,
                )
            end
        end
    end
end

close(io)
