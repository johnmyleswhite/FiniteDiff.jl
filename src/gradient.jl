"""
    gradient!{T <: AbstractFloat}(
        output::AbstractArray,
        f::Function,
        x::AbstractArray{T},
        buffer::AbstractArray{T},
        ::ForwardMode,
    )

# Description

Evaluate the gradient of `f` at `x` using forward-mode finite differences.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

# Arguments

* `output::AbstractArray`: An array that will be mutated to contain the
    gradient of `f` at `x`.
* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T}`: The value of `x` at which to evaluate the gradient of
    `f`.
* `buffer::AbstractArray{T}`: A buffer that is equivalent to `similar(x)`. Used
    as a temporary copy of `x` that can be mutated safely during the execution
    of the function.
* `::ForwardMode`: An instance of the `ForwardMode` type.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```julia
import FiniteDiff: gradient!, ForwardMode
x = [0.0, 0.0]
output, buffer = Array(Float64, 2), Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
    ForwardMode(),
)
```
"""
function gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
    ::ForwardMode,
)::Void
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
        x_i = x[i]

        # Temporarily change buffer[i] to a new value.
        ϵ = step_size(ForwardMode(), x_i)
        buffer[i] = x_i + ϵ

        # Evaluate f at the new value of x stored in the buffer.
        f_xp = f(buffer)

        # Restore the true value of x[i] in the buffer.
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        # TODO: Remove this
        # @printf("%d\t%s\t%s\t%s\t%s\n", i, f_xp, f_x, f_xp - f_x, ϵ)
        output[i] = (f_xp - f_x) / ϵ
    end

    # Return nothing. The result is already stored in output.
    return
end

"""
    gradient!{T <: AbstractFloat}(
        output::AbstractArray,
        f::Function,
        x::AbstractArray{T},
        buffer::AbstractArray{T},
        ::BackwardMode,
    )

# Description

Evaluate the gradient of `f` at `x` using backward-mode finite differences.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

# Arguments

* `output::AbstractArray`: An array that will be mutated to contain the
    gradient of `f` at `x`.
* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T}`: The value of `x` at which to evaluate the gradient of
    `f`.
* `buffer::AbstractArray{T}`: A buffer that is equivalent to `similar(x)`. Used
    as a temporary copy of `x` that can be mutated safely during the execution
    of the function.
* `::BackwardMode`: An instance of the `BackwardMode` type.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```julia
import FiniteDiff: gradient!, BackwardMode
x = [0.0, 0.0]
output, buffer = Array(Float64, 2), Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
    BackwardMode(),
)
```
"""
function gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
    ::BackwardMode,
)::Void
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

        # Evaluate f at the new value of x stored in the buffer.
        f_xm = f(buffer)

        # Restore the true value of x[i] in the buffer.
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = (f_x - f_xm) / ϵ
    end

    # Return nothing. The result is already stored in output.
    return
end

"""
    gradient!{T <: AbstractFloat}(
        output::AbstractArray,
        f::Function,
        x::AbstractArray{T},
        buffer::AbstractArray{T},
        ::CentralMode,
    )

# Description

Evaluate the gradient of `f` at `x` using central-mode finite differences.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

# Arguments

* `output::AbstractArray`: An array that will be mutated to contain the
    gradient of `f` at `x`.
* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T}`: The value of `x` at which to evaluate the gradient of
    `f`.
* `buffer::AbstractArray{T}`: A buffer that is equivalent to `similar(x)`. Used
    as a temporary copy of `x` that can be mutated safely during the execution
    of the function.
* `::CentralMode`: An instance of the `CentralMode` type.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```julia
import FiniteDiff: gradient!, CentralMode
x = [0.0, 0.0]
output, buffer = Array(Float64, 2), Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
    CentralMode(),
)
```
"""
function gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
    ::CentralMode,
)::Void
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

        # Evaluate f at the new value of x stored in the buffer.
        f_xp = f(buffer)

        # Temporarily change buffer[i] to a new value.
        buffer[i] = x_i - ϵ

        # Evaluate f at the new value of x stored in the buffer.
        f_xm = f(buffer)

        # Restore the true value of x[i] in the buffer.
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        # TODO: Remove this
        # @printf("%d\t%s\t%s\t%s\t%s\n", i, f_xp, f_xm, f_xp - f_xm, ϵ)
        output[i] = (f_xp - f_xm) / (ϵ + ϵ)
    end

    # Return nothing. The result is already stored in output.
    return
end

"""
    gradient!{T <: AbstractFloat}(
        output::AbstractArray,
        f::Function,
        x::AbstractArray{T},
        buffer::AbstractArray{Complex{T}},
        ::ComplexMode,
    )

# Description

Evaluate the gradient of `f` at `x` using complex-mode finite differences.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.

**NOTE**: The buffer must have complex elements to allow the computation of
complex-mode finite differences.

**NOTE**: This mode of finite differences will work correctly only when:

* `f` supports complex inputs.
* `f` is an analytic function in the complex analysis sense of the word.

# Arguments

* `output::AbstractArray`: An array that will be mutated to contain the
    gradient of `f` at `x`.
* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T}`: The value of `x` at which to evaluate the gradient of
    `f`.
* `buffer::AbstractArray{Complex{T}}`: A buffer that is equivalent to
    `similar(x, Complex{eltype(x)})`. Used as a temporary copy of `x` that can
    be mutated safely during the execution of the function.
* `::ComplexMode`: An instance of the `ComplexMode` type.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```julia
import FiniteDiff: gradient!, ComplexMode
x = [0.0, 0.0]
output, buffer = Array(Float64, 2), Array(Complex{Float64}, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
    ComplexMode(),
)
```
"""
function gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{Complex{T}},
    ::ComplexMode,
)::Void
    # Validate that all inputs have the same number of dimensions.
    n = length(x)
    if length(output) != n || length(buffer) != n
        throw(DomainError())
    end

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over dimensions of the input.
    for i in 1:n
        # Make a copy of the true value of x[i].
        x_i = x[i]

        # Temporarily change buffer[i] to a new value.
        ϵ = step_size(ComplexMode(), x_i)
        buffer[i] = x_i + ϵ * im

        # Evaluate f at the new value of x stored in the buffer.
        f_xim = f(buffer)

        # Restore the true value of x[i] in the buffer.
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = imag(f_xim) / ϵ
    end

    # Return nothing. The result is already stored in output.
    return
end

"""
    gradient!{T <: AbstractFloat}(
        output::AbstractArray,
        f::Function,
        x::AbstractArray{T},
        buffer::AbstractArray{T},
    )

# Description

Evaluate the gradient of `f` at `x` using finite differences. Defaults to
central mode finite differences. See the documentation for that mode for
more details.

# Arguments

* `output::AbstractArray`: An array that will be mutated to contain the
    gradient of `f` at `x`.
* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T}`: The value of `x` at which to evaluate the gradient of
    `f`.
* `buffer::AbstractArray{T}`: A buffer that is equivalent to `similar(x)`. Used
    as a temporary copy of `x` that can be mutated safely during the execution
    of the function.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```julia
import FiniteDiff: gradient!
x = [0.0, 0.0]
output, buffer = Array(Float64, 2), Array(Float64, 2)
gradient!(
    output,
    x -> sin(x[1]) + 2 * sin(x[2]),
    x,
    buffer,
)
```
"""
function gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
)::Void
    gradient!(output, f, x, buffer, CentralMode())
    return
end

"""
    gradient{T <: AbstractFloat}(
        f::Function,
        x::AbstractArray{T},
        mode::Mode = CentralMode(),
    )

# Description

Evaluate the gradient of `f` at `x` using finite differences. Defaults to
central mode finite differences. See the documentation for that mode for
more details.

**NOTE**: Allocates new memory to store the output as well as memory to use as
a buffer.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractArray{T}`: The value of `x` at which to evaluate the gradient of
    `f`.
* `m::Mode`: An instance of the `Mode` type. This will determine the mode
    of finite differences that will be used.

# Returns

* `output::AbstractArray`: The gradient.

# Examples

```julia
import FiniteDiff: gradient!, ForwardMode
x = [0.0, 0.0]
output = gradient!(x -> sin(x[1]) + 2 * sin(x[2]), x, ForwardMode())
```
"""
function gradient{T <: AbstractFloat}(
    f::Function,
    x::AbstractArray{T},
    mode::Mode = CentralMode(),
)::AbstractArray{T}
    if mode != ComplexMode()
        output, buffer = similar(x), similar(x)
    else
        output, buffer = similar(x), similar(x, Complex{T})
    end
    gradient!(output, f, x, buffer, mode)
    return output
end

"""
    gradient(f::Function, mode::Mode = CentralMode(); mutates::Bool = false)

# Description

Construct a new function that will evaluate the gradient of `f` at any value
of `x`. The user can specify the mode as a positional argument and can also
indicate whether to return a mutating or non-mutating function using a keyword
argument.

# Arguments

* `f::Function`: The function to be differentiated.
* `m::Mode`: An instance of the `Mode` type. This will determine the mode
    of finite differences that will be used.

# Keyword Arguments

* `mutates::Bool = false`: Determine whether the resulting function will mutate
    its inputs or will be a pure function.

# Returns

* `g′::Function`: A function to evaluate the gradient of `f` at a new point
    `x`.

# Examples

```julia
import FiniteDiff: gradient!, ForwardMode
x = [0.0, 0.0]
g′ = gradient(x -> sin(x[1]) + 2 * sin(x[2]), ForwardMode(), mutates = false)
output = g′(x)
```
"""
function gradient(
    f::Function,
    mode::Mode = CentralMode();
    mutates::Bool = false,
)::Function
    if mutates
        return (output, x, buffer) -> gradient!(output, f, x, buffer, mode)
    else
        return (x, ) -> gradient(f, x, mode)
    end
end
