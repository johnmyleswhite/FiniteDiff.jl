"""
Evaluate the second derivative of `f` at `x` using finite-differencing.
"""
@inline function second_derivative(f::Function, x::AbstractFloat)
    ϵ = step_size(HessianMode(), x)
    return (f(x + ϵ) - 2 * f(x) + f(x - ϵ)) / (ϵ * ϵ)
end

"""
Evaluate the second derivative of `f` at `x` using finite-differencing. Mutates
the first element of `output` so that it contains the result.
"""
function second_derivative!{T <: AbstractFloat}(
    output::Vector{T},
    f::Function,
    x::AbstractFloat,
)
    output[1] = second_derivative(f, x)
    return
end

"""
Construct a new function that will evaluate the second derivative of `f` at
any point `x`. A keyword argument specifies whether the function should be
mutating or non-mutating.
"""
function second_derivative(f::Function; mutates::Bool=false)
    if mutates
        return (output, x) -> second_derivative!(output, f, x)
    else
        return (x, ) -> second_derivative(f, x)
    end
end
