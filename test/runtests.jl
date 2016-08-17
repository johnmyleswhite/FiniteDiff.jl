using Base.Test

srand(1)

print_with_color(:blue, "Running tests:\n")

tests = (
    "modes.jl",
    "step_size.jl",
    "derivative.jl",
    "second_derivative.jl",
    "gradient.jl",
    "hessian.jl",
)

@testset "All FiniteDiff tests" begin
    for t in tests
        try
            include(t)
            print_with_color(:green, @sprintf("* %s\n", t))
        catch
            print_with_color(:red, @sprintf("* %s\n", t))
        end
    end
end
