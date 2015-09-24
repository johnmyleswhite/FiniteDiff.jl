# Mutating
function hessian!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Matrix{T},
    f::Function,
    x::Vector{S},
)
    n = length(x)

    f_x = f(x)
    for i = 1:n
        x_i = x[i]

        ϵ = @hessian(x_i)

        x[i] = x_i + ϵ
        f_xpp = f(x)

        x[i] = x_i - ϵ
        f_xmm = f(x)

        output[i, i] = (f_xpp - 2 * f_x + f_xmm) / (ϵ * ϵ)

        ϵ_i = @central(x_i)

        for j = (i + 1):n
            x_j = x[j]

            ϵ_j = @central(x_j)

            x[i] = x_i + ϵ_i

            x[j] = x_j + ϵ_j
            f_xpp = f(x)

            x[j] = x_j - ϵ_j
            f_xpm = f(x)

            x[i] = x_i - ϵ_i

            x[j] = x_j + ϵ_j
            f_xmp = f(x)

            x[j] = x_j - ϵ_j
            f_xmm = f(x)

            output[i, j] = (f_xpp - f_xpm - f_xmp + f_xmm) / (4.0 * ϵ_i * ϵ_j)

            x[j] = x_j
        end

        x[i] = x_i
    end
    Base.LinAlg.copytri!(output, 'U')
    nothing
end

# Non-mutating
function hessian{T <: AbstractFloat}(f::Function, x::Vector{T})
    n = length(x)
    output = Array(Float64, n, n)
    hessian!(output, f, x)
    output
end

# Higher-order function
function hessian(f::Function; mutates::Bool = false)
    if mutates
        f′′!(output, x) = hessian!(output, f, x)
        return f′′!
    else
        f′′(x) = hessian(f, x)
        return f′′
    end
end
