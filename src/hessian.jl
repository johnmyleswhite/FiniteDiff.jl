"""
    hessian!{T <: AbstractFloat}(
        output::AbstractArray,
        f::Function,
        x::AbstractArray{T},
        buffer::AbstractArray{T},
    )

# Description

Evaluate the Hessian of `f` at `x` using finite differences. Store the results
into `output`. Work with a user-provided `buffer` to ensure that we can work
without copies, but also without mutating `x`. In Rust jargon, we take
ownership of both `output` and `buffer`, but do not require ownership of `x`.
We just need read-only access to `x`.

# Arguments

* `output::AbstractArray`: An array that will be mutated to contain the
    gradient.
* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T}`: The value of `x` at which to evaluate the Hessian of
    `f`.
* `buffer::AbstractArray{T}`: A buffer that is equivalent to `similar(x)`. Used
    as a temporary copy of `x` that can be mutated.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```julia
import FiniteDiff: hessian!
x = [0.0, 0.0]
output, buffer = similar(x, 2, 2), similar(x, 2)
hessian!(output, x -> sin(x[1]) + 2 * sin(x[2]), x, buffer)
```
"""
function hessian!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
)::Void
    # Validate that all inputs have the expected number of dimensions.
    n = length(x)
    if size(output) != (n, n) || length(buffer) != n
        throw(DomainError())
    end

    # Cache the value of f(x) to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over the rows of the input.
    for i = 1:n
        # Make a copy of the true value of x[i].
        x_i = x[i]

        # Temporarily change buffer[i] to a new value and evaluate f at the
        # new x stored in the buffer.
        ϵ = step_size(HessianMode(), x_i)
        buffer[i] = x_i + ϵ
        f_xpp = f(buffer)

        # Temporarily change buffer[i] to a new value and evaluate f at the
        # new x stored in the buffer.
        buffer[i] = x_i - ϵ
        f_xmm = f(buffer)

        # Store the diagonal component of the Hessian into the output array.
        output[i, i] = (f_xpp - 2 * f_x + f_xmm) / (ϵ * ϵ)

        # Determine the step-size to use for this row of the Hessian on the
        # off-diagonal elements.
        ϵ_i = step_size(CentralMode(), x_i)

        # Iterate over the columns of the input for the current row.
        for j = (i + 1):n
            # Make a copy of the true value of x[i].
            x_j = x[j]

            # Determine the step-size to use for this entry of the Hessian.
            ϵ_j = step_size(CentralMode(), x_j)

            # Temporarily change buffer[i] and buffer[j] to new values and
            # evaluate f at the new values of x stored in the buffer.
            buffer[i] = x_i + ϵ_i
            buffer[j] = x_j + ϵ_j
            f_xpp = f(buffer)
            buffer[j] = x_j - ϵ_j
            f_xpm = f(buffer)

            # Temporarily change buffer[i] and buffer[j] to new values and
            # evaluate f at the new values of x stored in the buffer.
            buffer[i] = x_i - ϵ_i
            buffer[j] = x_j + ϵ_j
            f_xmp = f(buffer)
            buffer[j] = x_j - ϵ_j
            f_xmm = f(buffer)

            # Store the i, j-th component of the Hessian into the output array.
            t4 = convert(T, 4)
            output[i, j] = (f_xpp - f_xpm - f_xmp + f_xmm) / (t4 * ϵ_i * ϵ_j)

            # Restore the true value of x[j] in the buffer.
            buffer[j] = x_j
        end

        # Restore the true value of x[i] in the buffer.
        buffer[i] = x_i
    end

    # Copy the upper triangular values into the lower triangular values to
    # symmetrize the results.
    Base.LinAlg.copytri!(output, 'U')

    # Return nothing. The result is already stored in output.
    return
end

"""
    hessian{T <: AbstractFloat}(f::Function, x::AbstractArray{T})

# Description

Evaluate the Hessian of `f` at `x` using finite differences. See
`hessian!(f, x, buffer)` for more details.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T <: AbstractArray}`: The value of `x` at which to evaluate
    the Hessian of `f`.

# Returns

* `output::Array{T <: AbstractArray}`: The Hessian of `f` at `x`.

# Examples

```julia
import FiniteDiff: hessian
x = [0.0, 0.0]
H = hessian(x -> sin(x[1]) + 2 * sin(x[2]), x)
```
"""
function hessian{T <: AbstractFloat}(
    f::Function,
    x::AbstractArray{T},
)::AbstractArray{T}
    # Determine the number of dimensions of the input.
    n = length(x)

    # Allocate memory for both the output and the temporary buffer.
    output, buffer = similar(x, n, n), similar(x, n)

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

```julia
import FiniteDiff: hessian
h = hessian(x -> sin(x[1]) + 2 * sin(x[2]), mutates = false)
x = [0.0, 0.0]
h(x)
```
"""
function hessian(f::Function; mutates::Bool = false)::Function
    if mutates
        return (output, x, buffer) -> hessian!(output, f, x, buffer)
    else
        return (x, ) -> hessian(f, x)
    end
end
