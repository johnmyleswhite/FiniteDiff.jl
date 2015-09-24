# Mutating
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    ::Type{Forward},
)
    n = length(x)
    f_x = f(x)
    for i in 1:n
        x_i = x[i]
        ϵ = @forward(x_i)
        x[i] = x_i + ϵ
        f_xp = f(x)
        x[i] = x_i
        output[i] = (f_xp - f_x) / ϵ
    end
    nothing
end

function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    ::Type{Backward},
)
    n = length(x)
    f_x = f(x)
    for i in 1:n
        x_i = x[i]
        ϵ = @backward(x_i)
        x[i] = x_i - ϵ
        f_xm = f(x)
        x[i] = x_i
        output[i] = (f_x - f_xm) / ϵ
    end
    nothing
end

function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    ::Type{Central},
)
    n = length(x)
    for i in 1:n
        ϵ = @central(x[i])
        x_i = x[i]
        x[i] = x_i + ϵ
        f_xp = f(x)
        x[i] = x_i - ϵ
        f_xm = f(x)
        x[i] = x_i
        output[i] = (f_xp - f_xm) / (ϵ + ϵ)
    end
    nothing
end

function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
)
    gradient!(output, f, x, Central)
end

# Non-mutating
function gradient{T <: AbstractFloat}(
    f::Function,
    x::Vector{T},
    mode = Central,
)
    output = similar(x, Float64)
    gradient!(output, f, x, mode)
    output
end

# Higher-order function
function gradient(f::Function, mode = Central; mutates::Bool = false)
    if mutates
        f′!(output, x) = gradient!(output, f, x, mode)
        return f′!
    else
        f′(x) = gradient(f, x, mode)
        return f′
    end
end
