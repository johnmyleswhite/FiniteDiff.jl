"""
    hessian!{S <: AbstractFloat, T <: AbstractFloat}(
        output::Matrix{T},
        f::Function,
        x::Vector{S},
        buffer::Vector{S},
    )

# Description

Evaluate the Hessian of `f` at `x` using finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

# Arguments

* `output::Vector{S}`: An array that will be mutated to contain the gradient.
* `f::Function`: The function to be differentiated.
* `x::Vector{T}`: The value of `x` at which to evaluate the Hessian of `f`.
* `buffer::Vector{T}`: A buffer that is equivalent to `similar(x)`. Used for
    temporary mutation.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```jl
import FiniteDiff: hessian!
x = [0.0, 0.0]
output = Array(Float64, 2, 2)
buffer = Array(Float64, 2)
hessian!(output, x -> sin(x[1]) + 2 * sin(x[2]), x, buffer)
```
"""
function hessian!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Matrix{T},
    f::Function,
    x::Vector{S},
    buffer::Vector{S},
)
    # Validate that all inputs have the expected number of dimensions.
    n = length(x)
    if size(output) != (n, n) || length(buffer) != n
        throw(DomainError())
    end

    # Cache the value of f(x) to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over the dimensions of the input.
    for i = 1:n
        # TODO: Improve comments for this loop and the nested loop inside it.
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

            buffer[i] = x_i + ϵ_i
            buffer[j] = x_j + ϵ_j
            f_xpp = f(buffer)
            buffer[j] = x_j - ϵ_j
            f_xpm = f(buffer)

            buffer[i] = x_i - ϵ_i
            buffer[j] = x_j + ϵ_j
            f_xmp = f(buffer)
            buffer[j] = x_j - ϵ_j
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

    # Return nothing. The result is already stored in output.
    return
end

"""
    hessian{T <: AbstractFloat}(f::Function, x::Vector{T})

# Description

Evaluate the Hessian of `f` at `x` using finite-differencing.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::Vector{T}`: The value of `x` at which to evaluate the Hessian of `f`.

# Returns

* `output::Matrix{S}`: The Hessian of `f` at `x`.

# Examples

```jl
import FiniteDiff: hessian
x = [0.0, 0.0]
H = hessian(x -> sin(x[1]) + 2 * sin(x[2]), x)
```
"""
function hessian{T <: AbstractFloat}(f::Function, x::Vector{T})
    # Determine the number of dimensions of the input.
    n = length(x)

    # Allocate memory for both the output and the temporary buffer.
    output = Array(Float64, n, n)
    buffer = Array(Float64, n)

    # Compute the Hessian using the mutating variant of this function.
    hessian!(output, f, x, buffer)

    # Return the output, but discard the temporary buffer.
    return output
end

"""
    hessian(f::Function; mutates::Bool = false)

# Description

Construct a new function that will evaluate the Hessian of `f` at any value
of `x`. The user can whether to return a mutating or non-mutating function
using a keyword argument.

# Arguments

* `f::Function`: The function to be differentiated.

# Keyword Arguments

* `mutates::Bool = false`: Determine whether the resulting function will mutate
    its inputs or will be a pure function.

# Returns

* `h::Function`: A function that will evaluate the Hessian of `f` at any `x`.

# Examples

```jl
import FiniteDiff: hessian
h = hessian(x -> sin(x[1]) + 2 * sin(x[2]))
```
"""
function hessian(f::Function; mutates::Bool = false)
    # TODO: Decide whether to allocate a buffer for users automatically.
    #    Doing so would require that users provide x up-front so we know its
    #    size.
    if mutates
        return (output, x, buffer) -> hessian!(output, f, x, buffer)
    else
        return (x, ) -> hessian(f, x)
    end
end
