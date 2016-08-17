"""
Evaluate the gradient of `f` at `x` using forward-mode finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.
"""
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    buffer::Vector{T},
    ::ForwardMode,
)
    # TODO: Raise appropriate errors here when these assertions are false.
    n = length(x)
    @assert n == length(output)
    @assert n == length(buffer)

    # Cache the value of f at x to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over dimensions of the input.
    for i in 1:n
        # Make a copy of the true value of x[i].
        x_i = buffer[i]

        # Temporarily change x[i] to a new value.
        ϵ = step_size(ForwardMode(), x_i)
        buffer[i] = x_i + ϵ

        # Evaluate f at the new value of x.
        f_xp = f(buffer)

        # Restore the true value of x[].
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = (f_xp - f_x) / ϵ
    end

    return
end

"""
Evaluate the gradient of `f` at `x` using backward-mode finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.
"""
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    buffer::Vector{T},
    ::BackwardMode,
)
    # TODO: Raise appropriate errors here when these assertions are false.
    n = length(x)
    @assert n == length(output)
    @assert n == length(buffer)

    # Cache the value of f at x to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over dimensions of the input.
    for i in 1:n
        # Make a copy of the true value of x[i].
        x_i = buffer[i]

        # Temporarily change x[i] to a new value.
        ϵ = step_size(BackwardMode(), x_i)
        buffer[i] = x_i - ϵ

        # Evaluate f at the new value of x.
        f_xm = f(buffer)

        # Restore the true value of x[].
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = (f_x - f_xm) / ϵ
    end

    return
end

"""
Evaluate the gradient of `f` at `x` using central-mode finite-differencing.
Store the results into `output`. Work with a user-provided `buffer` to ensure
that we can work without copies, but also without mutating `x`. In Rust jargon,
we take ownership of both `output` and `buffer`, but do not require ownership
of `x`. We just need read-only access to `x`.
"""
function gradient!{S <: AbstractFloat, T <: AbstractFloat}(
    output::Vector{S},
    f::Function,
    x::Vector{T},
    buffer::Vector{T},
    ::CentralMode,
)
    # TODO: Raise appropriate errors here when these assertions are false.
    n = length(x)
    @assert n == length(output)
    @assert n == length(buffer)

    # Cache the value of f at x to avoid multiple calls to f, which is pure.
    f_x = f(x)

    # Copy x into the user-supplied buffer.
    copy!(buffer, x)

    # Iterate over dimensions of the input.
    for i in 1:n
        # Make a copy of the true value of x[i].
        x_i = buffer[i]

        # Temporarily change x[i] to a new value.
        ϵ = step_size(CentralMode(), x_i)
        buffer[i] = x_i + ϵ

        # Evaluate f at the new value of x.
        f_xp = f(buffer)

        # Temporarily change x[i] to a new value.
        buffer[i] = x_i - ϵ

        # Evaluate f at the new value of x.
        f_xm = f(buffer)

        # Restore the true value of x[].
        buffer[i] = x_i

        # Store the i-th component of the gradient into the output array.
        output[i] = (f_xp - f_xm) / (ϵ + ϵ)
    end

    return
end

"""
Evaluate the gradient of `f` at `x` using finite-differencing. Defaults to
central mode finite-differencing. See the documentation for that mode for
more details.
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
Evaluate the gradient of `f` at `x` using finite-differencing. Defaults to
central mode finite-differencing. See the documentation for that mode for
more details.

NOTE: Allocates new memory to store the output as well as memory to use as a
buffer.
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
Construct a new function that will evaluate the gradient of `f` at any value
of `x`. The user can specify the mode as a positional argument and can also
indicate whether to return a mutating or non-mutating function using a keyword
argument.
"""
function gradient(f::Function, mode = CentralMode(); mutates::Bool = false)
    # TODO: Decide whether to allocate a buffer for users automatically.
    if mutates
        return (output, x, buffer) -> gradient!(output, f, x, buffer, mode)
    else
        return (x, ) -> gradient(f, x, mode)
    end
end
