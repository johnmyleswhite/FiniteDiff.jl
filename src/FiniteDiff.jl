isdefined(Base, :__precompile__) && __precompile__(true)

module FiniteDiff
    include("eps.jl")
    include("modes.jl")
    include("derivative.jl")
    include("gradient.jl")
    include("second_derivative.jl")
    include("hessian.jl")
end
