# Mutating
function derivative!{T <: AbstractFloat}(
    output::Vector{T},
    f::Function,
    x::AbstractFloat,
    ::Type{Forward},
)
    ϵ = @forward(x)
    output[1] = (f(x + ϵ) - f(x)) / ϵ
end

function derivative!{T <: AbstractFloat}(
    output::Vector{T},
    f::Function,
    x::AbstractFloat,
    ::Type{Backward},
)
    ϵ = @backward(x)
    output[1] = (f(x) - f(x - ϵ)) / ϵ
end

function derivative!{T <: AbstractFloat}(
    output::Vector{T},
    f::Function,
    x::AbstractFloat,
    ::Type{Central},
)
    ϵ = @central(x)
    output[1] = (f(x + ϵ) - f(x - ϵ)) / (ϵ + ϵ)
end

function derivative!{T <: AbstractFloat}(
    output::Vector{T},
    f::Function,
    x::AbstractFloat,
    ::Type{Complex},
)
    ϵ = @complex(x)
    output[1] = imag(f(x + ϵ * im)) / ϵ
end

function derivative!{T <: AbstractFloat}(
    output::Vector{T},
    f::Function,
    x::AbstractFloat,
)
    derivative!(output, f, x, Central)
end

# Non-mutating
function derivative(f::Function, x::AbstractFloat, ::Type{Forward})
    ϵ = @forward(x)
    (f(x + ϵ) - f(x)) / ϵ
end

function derivative(f::Function, x::AbstractFloat, ::Type{Backward})
    ϵ = @backward(x)
    (f(x) - f(x - ϵ)) / ϵ
end

function derivative(f::Function, x::AbstractFloat, ::Type{Central})
    ϵ = @central(x)
    (f(x + ϵ) - f(x - ϵ)) / (ϵ + ϵ)
end

function derivative(f::Function, x::AbstractFloat, ::Type{Complex})
    ϵ = @complex(x)
    imag(f(x + ϵ * im)) / ϵ
end

derivative(f::Function, x::AbstractFloat) = derivative(f, x, Central)

# Higher-order function
function derivative(f::Function, mode = Central; mutates::Bool=false)
    if mutates
        f′!(output, x) = derivative!(output, f, x, mode)
        return f′!
    else
        f′(x) = derivative(f, x, mode)
        return f′
    end
end
