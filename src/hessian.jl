"""
Evaluate the Hessian of `f` at `x` using finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.
"""
function hessian!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Matrix{T},
    f::Function,
    x::Vector{S},
    buffer::Vector{S},
)
    # TODO: Improve comments.
    n = length(x)

    f_x = f(x)

    copy!(buffer, x)

    for i = 1:n
        x_i = x[i]

        ϵ = step_size(HessianMode(), x_i)

        buffer[i] = x_i + ϵ
        f_xpp = f(buffer)

        buffer[i] = x_i - ϵ
        f_xmm = f(buffer)

        output[i, i] = (f_xpp - 2 * f_x + f_xmm) / (ϵ * ϵ)

        ϵ_i = step_size(CentralMode(), x_i)

        for j = (i + 1):n
            x_j = x[j]

            ϵ_j = step_size(CentralMode(), x_j)

            buffer[i], buffer[j] = x_i + ϵ_i, x_j + ϵ_j
            f_xpp = f(buffer)

            buffer[i], buffer[j] = x_i + ϵ_i, x_j - ϵ_j
            f_xpm = f(buffer)

            buffer[i], buffer[j] = x_i - ϵ_i, x_j + ϵ_j
            f_xmp = f(buffer)

            buffer[i], buffer[j] = x_i - ϵ_i, x_j - ϵ_j
            f_xmm = f(buffer)

            s4 = convert(S, 4)
            output[i, j] = (f_xpp - f_xpm - f_xmp + f_xmm) / (s4 * ϵ_i * ϵ_j)

            buffer[j] = x_j
        end

        buffer[i] = x_i
    end

    # Copy the upper triangular values into the lower triangular values to
    # symmetrize the results.
    Base.LinAlg.copytri!(output, 'U')

    return
end

"""
Evaluate the Hessian of `f` at `x` using finite-differencing.
"""
function hessian{T <: AbstractFloat}(f::Function, x::Vector{T})
    n = length(x)
    output = Array(Float64, n, n)
    buffer = Array(Float64, n)
    hessian!(output, f, x, buffer)
    return output
end

"""
Construct a new function that will evaluate the Hessian of `f` at any value
of `x`. The user can whether to return a mutating or non-mutating function
using a keyword argument.
"""
function hessian(f::Function; mutates::Bool = false)
    if mutates
        return (output, x, buffer) -> hessian!(output, f, x, buffer)
    else
        return (x, ) -> hessian(f, x)
    end
end
