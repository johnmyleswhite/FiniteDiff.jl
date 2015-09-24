# Mutating
function second_derivative!{T <: AbstractFloat}(
    output::Vector{T},
    f::Function,
    x::AbstractFloat
)
    ϵ = @hessian(x)
    output[1] = (f(x + ϵ) - 2 * f(x) + f(x - ϵ)) / (ϵ * ϵ)
end

# Non-mutating
function second_derivative(f::Function, x::AbstractFloat)
    ϵ = @hessian(x)
    (f(x + ϵ) - 2 * f(x) + f(x - ϵ)) / (ϵ * ϵ)
end

# Higher-order function
function second_derivative(f::Function; mutates::Bool=false)
    if mutates
        f′!(output, x) = second_derivative!(output, f, x)
        return f′!
    else
        f′(x) = second_derivative(f, x)
        return f′
    end
end
