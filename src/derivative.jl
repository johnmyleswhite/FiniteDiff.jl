doc"""
    derivative(f::Function, x::AbstractFloat, ::ForwardMode)

# Description

Evaluate the derivative of `f` at `x` using forward finite-differencing. In
mathematical notation, we calculate,

```math
\frac{f(x + \epsilon) - f(x)}{\epsilon},
```

where ``\epsilon`` is chosen to be small enough to approximate the derivative,
but not so small as to suffer from extreme numerical inaccuracy.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.
* `::ForwardMode`: An instance of the `ForwardMode` type.

# Returns

* `y::Real`: The derivative of `f` evaluated at `x`.

# Examples

```jl
import FiniteDiff: derivative, ForwardMode
y = derivative(sin, 0.0, ForwardMode())
```
"""
@inline function derivative(f::Function, x::AbstractFloat, ::ForwardMode)::Real
    ϵ = step_size(ForwardMode(), x)
    return (f(x + ϵ) - f(x)) / ϵ
end

doc"""
    derivative(f::Function, x::AbstractFloat, ::BackwardMode)

# Description

Evaluate the derivative of `f` at `x` using backward finite-differencing. In
mathematical notation, we calculate,

```math
\frac{f(x) - f(x - \epsilon)}{\epsilon},
```

where ``\epsilon`` is chosen to be small enough to approximate the derivative,
but not so small as to suffer from extreme numerical inaccuracy.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.
* `::BackwardMode`: An instance of the `BackwardMode` type.

# Returns

* `y::Real`: The derivative of `f` evaluated at `x`.

# Examples

```jl
import FiniteDiff: derivative, BackwardMode
y = derivative(sin, 0.0, BackwardMode())
```
"""
@inline function derivative(f::Function, x::AbstractFloat, ::BackwardMode)::Real
    ϵ = step_size(BackwardMode(), x)
    return (f(x) - f(x - ϵ)) / ϵ
end

doc"""
    derivative(f::Function, x::AbstractFloat, ::CentralMode)

# Description

Evaluate the derivative of `f` at `x` using central finite-differencing. In
mathematical notation, we calculate,

```math
\frac{f(x + \epsilon) - f(x - \epsilon)}{2 \epsilon},
```

where ``\epsilon`` is chosen to be small enough to approximate the derivative,
but not so small as to suffer from extreme numerical inaccuracy.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.
* `::CentralMode`: An instance of the `CentralMode` type.

# Returns

* `y::Real`: The derivative of `f` evaluated at `x`.

# Examples

```jl
import FiniteDiff: derivative, CentralMode
y = derivative(sin, 0.0, CentralMode())
```
"""
@inline function derivative(f::Function, x::AbstractFloat, ::CentralMode)::Real
    ϵ = step_size(CentralMode(), x)
    return (f(x + ϵ) - f(x - ϵ)) / (ϵ + ϵ)
end

doc"""
    derivative(f::Function, x::AbstractFloat, ::ComplexMode)

# Description

Evaluate the derivative of `f` at `x` using complex finite-differencing. In
mathematical notation, we calculate,

```math
\frac{\operatorname{Im}(f(x + \epsilon i))}{\epsilon}
```

where ``\epsilon`` is chosen to be as small as possible.

**NOTE**: This mode of finite-differencing will work correctly only when:

* `f` supports complex inputs.
* `f` is an analytic function in the complex analysis sense of the word.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.
* `::ForwardMode`: An instance of the `ForwardMode` type.

# Returns

* `y::Real`: The derivative of `f` evaluated at `x`.

# Examples

```jl
import FiniteDiff: derivative, ComplexMode
y = derivative(sin, 0.0, ComplexMode())
```
"""
@inline function derivative(f::Function, x::AbstractFloat, ::ComplexMode)::Real
    ϵ = step_size(ComplexMode(), x)
    return imag(f(x + ϵ * im)) / ϵ
end

"""
    derivative!(
        output::AbstractArray,
        f::Function,
        x::AbstractFloat,
        m::Mode = CentralMode(),
    )

# Description

Evaluate the derivative of `f` at `x` using finite-differencing. The first
argument `output` will be mutated so that its first element will contain the
result.

See the documentation for the non-mutating versions of `derivative` for
additional information about the effects of choosing a specific mode of
finite-differencing.

# Arguments

* `output::AbstractArray`: An array whose first element will be mutated.
* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.
* `m::Mode`: An instance of the `Mode` type. This will determine the mode
    of finite-differencing that will be used.

# Returns

* `nothing::Void`: This function is called for its side effects.

# Examples

```jl
import FiniteDiff: derivative, CentralMode
y = Array(Float64, 1)
derivative!(y, sin, 0.0, CentralMode())
```
"""
function derivative!(
    output::AbstractArray,
    f::Function,
    x::AbstractFloat,
    m::Mode = CentralMode(),
)::Void
    output[1] = derivative(f, x, m)
    return
end

"""
    derivative(f::Function, x::AbstractFloat)

# Description

Evaluate the derivative of `f` at `x` using finite-differencing without having
to specify the mode of finite-differencing. Currently defaults to using central
finite-differencing, which is equivalent to the user calling
`derivative(f, x, CentralMode())` instead of `derivative(f, x)`. See the
documentation for `derivative(f, x, CentralMode())` for more details.

# Arguments

* `f::Function`: The function to be differentiated.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `y::Real`: The derivative of `f` evaluated at `x`.

# Examples

```jl
import FiniteDiff: derivative
y = derivative(sin, 0.0)
```
"""
function derivative(f::Function, x::AbstractFloat)::Real
    return derivative(f, x, CentralMode())
end

"""
    derivative(f::Function, mode::Mode = CentralMode(); mutates::Bool=false)

# Description

Construct a new function that will evaluate the derivative of `f` at any value
of `x` using finite-differencing. The user can specify the mode of
finite-differencing to use by providing a second positional argument. The user
can also indicate whether to return a mutating or non-mutating function using
the keyword argument, `mutates`, which should be either `true` or `false`.

# Arguments

* `f::Function`: The function to be differentiated.
* `m::Mode`: An instance of the `Mode` type. This will determine the mode
    of finite-differencing that will be used.

# Keyword Arguments

* `mutates::Bool = false`: Determine whether the resulting function will mutate
    its inputs or will be a pure function.

# Examples

```jl
import FiniteDiff: derivative
f′ = derivative(sin)
f′(0.0)
```
"""
function derivative(
    f::Function,
    mode::Mode = CentralMode();
    mutates::Bool=false
)::Function
    if mutates
        return (output, x) -> derivative!(output, f, x, mode)
    else
        return (x, ) -> derivative(f, x, mode)
    end
end
