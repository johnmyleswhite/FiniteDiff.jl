__precompile__(true)

module FiniteDiff
    include("modes.jl")
    include("step_size_v0.jl")
    # include("step_size_v1.jl")
    # include("step_size_v2.jl")
    # include("step_size_v3.jl")
    include("derivative.jl")
    include("second_derivative.jl")
    include("gradient.jl")
    include("hessian.jl")
end
