"""
    second_derivative(f::Function, x::AbstractFloat)

# Description

Evaluate the second derivative of `f` at `x` using finite-differencing. In
mathematical notation, we calculate,

```math
\frac{f(x + \epsilon) - 2 f(x) + f(x - \epsilon)}{\epsilon^2},
```

where ``\epsilon`` is chosen to be small enough to approximate the second
derivative, but not so small as to suffer from extreme numerical inaccuracy.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `y::Real`: The second derivative of `f` evaluated at `x`.

# Examples

```jl
import FiniteDiff: second_derivative
y = second_derivative(sin, 1.0)
```
"""
@inline function second_derivative(f::Function, x::AbstractFloat)
    ϵ = step_size(HessianMode(), x)
    return (f(x + ϵ) - 2 * f(x) + f(x - ϵ)) / (ϵ * ϵ)
end

"""
    second_derivative!{T <: AbstractFloat}(
        output::Vector{T},
        f::Function,
        x::AbstractFloat,
    )

# Description

Evaluate the second derivative of `f` at `x` using finite-differencing. In
mathematical notation, we calculate,

```math
\frac{f(x + \epsilon) - 2 f(x) + f(x - \epsilon)}{\epsilon^2},
```

where ``\epsilon`` is chosen to be small enough to approximate the second
derivative, but not so small as to suffer from extreme numerical inaccuracy.

The first argument, `output`, will be mutated so that the value of the second
derivative in its first element.

# Arguments

* `output::AbstractArray{T <: AbstractFloat}`: An vector whose first element
    will be mutated.
* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `void::Nothing`: This function is called for its side-effects.

# Examples

```jl
import FiniteDiff: second_derivative
y = Array(Float64, 1)
second_derivative!(y, sin, 1.0)
```
"""
function second_derivative!{T <: AbstractFloat}(
    output::AbstractArray{T},
    f::Function,
    x::AbstractFloat,
)
    output[1] = second_derivative(f, x)
    return
end

"""
    second_derivative(f::Function; mutates::Bool=false)

# Description

Construct a new function that will evaluate the second derivative of `f` at
any point `x`. A keyword argument specifies whether the function should be
mutating or non-mutating.

# Arguments

* `f::Function`: The function to be differentiated.

# Keyword Arguments

* `mutates::Bool = false`: Determine whether the resulting function will mutate
    its inputs or will be a pure function.

# Returns

* `void::Nothing`: This function is called for its side-effects.

# Examples

```jl
import FiniteDiff: second_derivative
f′′ = second_derivative(sin)
f′′(1.0)
```
"""
function second_derivative(f::Function; mutates::Bool=false)
    if mutates
        return (output, x) -> second_derivative!(output, f, x)
    else
        return (x, ) -> second_derivative(f, x)
    end
end
