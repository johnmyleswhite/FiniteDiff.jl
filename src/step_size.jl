"""
    step_size(::ForwardMode, x::AbstractFloat)

# Description

Determine the step-size to use for forward-mode finite-differencing of
gradients.

# Arguments

* `::ForwardMode`: An instance of the `ForwardMode` type.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite-differencing at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `sqrt(eps(x))`.

# Examples

```jl
import FiniteDiff: step_size, ForwardMode
ϵ = step_size(ForwardMode(), 0.0)
```
"""
@inline function step_size(::ForwardMode, x::AbstractFloat)::Real
    return sqrt(eps(x)) * max(one(x), abs(x))
end

"""
    step_size(::BackwardMode, x::AbstractFloat)

# Description

Determine the step-size to use for backward-mode finite-differencing of
gradients.

# Arguments

* `::BackwardMode`: An instance of the `BackwardMode` type.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite-differencing at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `sqrt(eps(x))`.

# Examples

```jl
import FiniteDiff: step_size, BackwardMode
ϵ = step_size(BackwardMode(), 0.0)
```
"""
@inline function step_size(::BackwardMode, x::AbstractFloat)::Real
    return sqrt(eps(x)) * max(one(x), abs(x))
end

"""
    step_size(::CentralMode, x::AbstractFloat)

# Description

Determine the step-size to use for central-mode finite-differencing of
gradients.

# Arguments

* `::CentralMode`: An instance of the `CentralMode` type.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite-differencing at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `cbrt(eps(x))`.

# Examples

```jl
import FiniteDiff: step_size, CentralMode
ϵ = step_size(CentralMode(), 0.0)
```
"""
@inline function step_size(::CentralMode, x::AbstractFloat)::Real
    return cbrt(eps(x)) * max(one(x), abs(x))
end

"""
    step_size(::ComplexMode, x::AbstractFloat)

# Description

Determine the step-size to use for complex-mode finite-differencing of
gradients.

# Arguments

* `::ComplexMode`: An instance of the `ComplexMode` type.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite-differencing at `x`.

# References

See "The Complex-Step Derivative Approximation" by Martins, Sturdza and Alonso
(2003) for the mathematical justification for working with `eps(x)`.

# Examples

```jl
import FiniteDiff: step_size, ComplexMode
ϵ = step_size(ComplexMode(), 0.0)
```
"""
@inline function step_size(::ComplexMode, x::AbstractFloat)::Real
    return eps(x)
end

"""
    step_size(::HessianMode, x::AbstractFloat)

# Description

Determine the step-size to use for finite-differencing of hessians.

# Arguments

* `::HessianMode`: An instance of the `HessianMode` type.
* `x::AbstractFloat`: An `AbstractFloat` value. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite-differencing at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `eps(x)^(1 // 4)`.

# Examples

```jl
import FiniteDiff: step_size, HessianMode
ϵ = step_size(HessianMode(), 0.0)
```
"""
@inline function step_size(::HessianMode, x::AbstractFloat)::Real
    return eps(x)^(1 // 4) * max(one(x), abs(x))
end
