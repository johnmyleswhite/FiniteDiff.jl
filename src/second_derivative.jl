doc"""
    second_derivative(f::Function, x::AbstractFloat)

# Description

Evaluate the second derivative of `f` at `x` using finite differences. In
mathematical notation, we calculate,

```math
\frac{f(x + \epsilon) - 2 f(x) + f(x - \epsilon)}{\epsilon^2},
```

where ``\epsilon`` is chosen to be small enough to approximate the second
derivative, but not so small as to suffer from extreme numerical inaccuracy.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: The point at which to evaluate the derivative of `f`. Its
    type must implement `eps`.

# Returns

* `y::Real`: The second derivative of `f` evaluated at `x`.

# Examples

```jl
import FiniteDiff: second_derivative
y = second_derivative(sin, 1.0)
```
"""
@inline function second_derivative(f::Function, x::AbstractFloat)::Real
    ϵ = step_size(HessianMode(), x)
    return (f(x + ϵ) - 2 * f(x) + f(x - ϵ)) / (ϵ * ϵ)
end

doc"""
    second_derivative!(
        output::AbstractArray,
        f::Function,
        x::AbstractFloat,
    )

# Description

Evaluate the second derivative of `f` at `x` using finite differences. In
mathematical notation, we calculate,

```math
\frac{f(x + \epsilon) - 2 f(x) + f(x - \epsilon)}{\epsilon^2},
```

where ``\epsilon`` is chosen to be small enough to approximate the second
derivative, but not so small as to suffer from extreme numerical inaccuracy.

The first argument, `output`, will be mutated so that the value of the second
derivative is its first element.

# Arguments

* `output::AbstractArray`: An array whose first element will be mutated.
* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: The point at which to evaluate the derivative of `f`. Its
    type must implement `eps`.

# Returns

* `void::Nothing`: This function is called for its side-effects.

# Examples

```jl
import FiniteDiff: second_derivative
y = Array(Float64, 1)
second_derivative!(y, sin, 1.0)
```
"""
function second_derivative!(
    output::AbstractArray,
    f::Function,
    x::AbstractFloat,
)::Void
    output[1] = second_derivative(f, x)
    return
end

"""
    second_derivative(f::Function; mutates::Bool=false)

# Description

Construct a new function that will evaluate the second derivative of `f` at
any point `x`. A keyword argument specifies whether the resulting function
should be mutating or non-mutating.

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
function second_derivative(f::Function; mutates::Bool=false)::Function
    if mutates
        return (output, x) -> second_derivative!(output, f, x)
    else
        return (x, ) -> second_derivative(f, x)
    end
end
