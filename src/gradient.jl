"""
    gradient!{S <: AbstractFloat, T <: AbstractFloat}(
        output::Vector{S},
        f::Function,
        x::Vector{T},
        buffer::Vector{T},
        ::ForwardMode,
    )

# Description

Evaluate the gradient of `f` at `x` using forward-mode finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

# Arguments

* `output::Vector{S}`: An array that will be mutated to contain the gradient.
* `f::Function`: The function to be differentiated.
* `x::Vector{T}`: The value of `x` at which to evaluate the gradient of `f`.
* `buffer::Vector{T}`: A buffer that is equivalent to `similar(x)`. Used for
    temporary mutation.
* `::ForwardMode`: An instance of the `ForwardMode` type.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```jl
import FiniteDiff: gradient!, ForwardMode
x = [0.0, 0.0]
output = Array(Float64, 2)
buffer = Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
    ForwardMode(),
)
```
"""
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    buffer::Vector{T},
    ::ForwardMode,
)
    # Validate that all inputs have the same number of dimensions.
    n = length(x)
    if length(output) != n || length(buffer) != n
        throw(DomainError())
    end

    # Cache the value of f(x) to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over the dimensions of the input.
    for i in 1:n
        # Make a copy of the true value of x[i].
        x_i = buffer[i]

        # Temporarily change buffer[i] to a new value.
        ϵ = step_size(ForwardMode(), x_i)
        buffer[i] = x_i + ϵ

        # Evaluate f at the new value of buffer.
        f_xp = f(buffer)

        # Restore the true value of buffer[i].
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = (f_xp - f_x) / ϵ
    end

    # Return nothing. The result is already stored in output.
    return
end

"""
    gradient!{S <: AbstractFloat, T <: AbstractFloat}(
        output::Vector{S},
        f::Function,
        x::Vector{T},
        buffer::Vector{T},
        ::BackwardMode,
    )

# Description

Evaluate the gradient of `f` at `x` using backward-mode finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

# Arguments

* `output::Vector{S}`: An array that will be mutated to contain the gradient.
* `f::Function`: The function to be differentiated.
* `x::Vector{T}`: The value of `x` at which to evaluate the gradient of `f`.
* `buffer::Vector{T}`: A buffer that is equivalent to `similar(x)`. Used for
    temporary mutation.
* `::BackwardMode`: An instance of the `BackwardMode` type.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```jl
import FiniteDiff: gradient!, BackwardMode
x = [0.0, 0.0]
output = Array(Float64, 2)
buffer = Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
    BackwardMode(),
)
```
"""
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    buffer::Vector{T},
    ::BackwardMode,
)
    # Validate that all inputs have the same number of dimensions.
    n = length(x)
    if length(output) != n || length(buffer) != n
        throw(DomainError())
    end

    # Cache the value of f(x) to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over dimensions of the input.
    for i in 1:n
        # Make a copy of the true value of x[i].
        x_i = buffer[i]

        # Temporarily change buffer[i] to a new value.
        ϵ = step_size(BackwardMode(), x_i)
        buffer[i] = x_i - ϵ

        # Evaluate f at the new value of buffer.
        f_xm = f(buffer)

        # Restore the true value of buffer[i].
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = (f_x - f_xm) / ϵ
    end

    # Return nothing. The result is already stored in output.
    return
end

"""
    gradient!{S <: AbstractFloat, T <: AbstractFloat}(
        output::Vector{S},
        f::Function,
        x::Vector{T},
        buffer::Vector{T},
        ::CentralMode,
    )

# Description

Evaluate the gradient of `f` at `x` using central-mode finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

# Arguments

* `output::Vector{S}`: An array that will be mutated to contain the gradient.
* `f::Function`: The function to be differentiated.
* `x::Vector{T}`: The value of `x` at which to evaluate the gradient of `f`.
* `buffer::Vector{T}`: A buffer that is equivalent to `similar(x)`. Used for
    temporary mutation.
* `::CentralMode`: An instance of the `CentralMode` type.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```jl
import FiniteDiff: gradient!, CentralMode
x = [0.0, 0.0]
output = Array(Float64, 2)
buffer = Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
    CentralMode(),
)
```
"""
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    buffer::Vector{T},
    ::CentralMode,
)
    # Validate that all inputs have the same number of dimensions.
    n = length(x)
    if length(output) != n || length(buffer) != n
        throw(DomainError())
    end

    # Cache the value of f(x) to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over dimensions of the input.
    for i in 1:n
        # Make a copy of the true value of x[i].
        x_i = buffer[i]

        # Temporarily change buffer[i] to a new value.
        ϵ = step_size(CentralMode(), x_i)
        buffer[i] = x_i + ϵ

        # Evaluate f at the new value of buffer.
        f_xp = f(buffer)

        # Temporarily change buffer[i] to a new value.
        buffer[i] = x_i - ϵ

        # Evaluate f at the new value of buffer.
        f_xm = f(buffer)

        # Restore the true value of buffer[i].
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = (f_xp - f_xm) / (ϵ + ϵ)
    end

    # Return nothing. The result is already stored in output.
    return
end

"""
    gradient!{S <: AbstractFloat, T <: AbstractFloat}(
        output::Vector{S},
        f::Function,
        x::Vector{T},
        buffer::Vector{T},
    )

# Description

Evaluate the gradient of `f` at `x` using finite-differencing. Defaults to
central mode finite-differencing. See the documentation for that mode for
more details.

# Arguments

* `output::Vector{S}`: An array that will be mutated to contain the gradient.
* `f::Function`: The function to be differentiated.
* `x::Vector{T}`: The value of `x` at which to evaluate the gradient of `f`.
* `buffer::Vector{T}`: A buffer that is equivalent to `similar(x)`. Used for
    temporary mutation.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```jl
import FiniteDiff: gradient!
x = [0.0, 0.0]
output = Array(Float64, 2)
buffer = Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
)
```
"""
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    buffer::Vector{T},
)
    gradient!(output, f, x, buffer, CentralMode())
    return
end

"""
    gradient{T <: AbstractFloat}(
        f::Function,
        x::Vector{T},
        mode = CentralMode(),
    )

# Description

Evaluate the gradient of `f` at `x` using finite-differencing. Defaults to
central mode finite-differencing. See the documentation for that mode for
more details.

**NOTE**: Allocates new memory to store the output as well as memory to use as
a buffer.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::Vector{T}`: The value of `x` at which to evaluate the gradient of `f`.
* `m::Mode`: An instance of the `Mode` type. This will determine the mode
    of finite-differencing that will be used.

# Returns

* `output::AbstractArray`: The gradient.

# Examples

```jl
import FiniteDiff: gradient!, ForwardMode
x = [0.0, 0.0]
gradient!(
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    ForwardMode(),
)
```
"""
function gradient{T <: AbstractFloat}(
    f::Function,
    x::Vector{T},
    mode = CentralMode(),
)
    output, buffer = similar(x, T), similar(x, T)
    gradient!(output, f, x, buffer, mode)
    return output
end

"""
    gradient(f::Function, mode = CentralMode(); mutates::Bool = false)

# Description

Construct a new function that will evaluate the gradient of `f` at any value
of `x`. The user can specify the mode as a positional argument and can also
indicate whether to return a mutating or non-mutating function using a keyword
argument.

# Arguments

* `f::Function`: The function to be differentiated.
* `m::Mode`: An instance of the `Mode` type. This will determine the mode
    of finite-differencing that will be used.

# Keyword Arguments

* `mutates::Bool = false`: Determine whether the resulting function will mutate
    its inputs or will be a pure function.

# Returns

* `g′::Function`: A function to evaluate the gradient of `f` at a new point `x`.

# Examples

```jl
import FiniteDiff: gradient!, ForwardMode
gradient(x -> sin(x[1]) + 2 * sin(x[2]), ForwardMode())
```
"""
function gradient(f::Function, mode = CentralMode(); mutates::Bool = false)
    # TODO: Decide whether to allocate a buffer for users automatically.
    #    Doing so would require that users provide x up-front so we know its
    #    size.
    if mutates
        return (output, x, buffer) -> gradient!(output, f, x, buffer, mode)
    else
        return (x, ) -> gradient(f, x, mode)
    end
end
