__precompile__(true)

module FiniteDiff
    include("modes.jl")
    include("step_size.jl")
    include("derivative.jl")
    include("second_derivative.jl")
    include("gradient.jl")
    include("hessian.jl")
end
