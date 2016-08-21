"""
    step_size(::ForwardMode, x::AbstractFloat)

# Description

Determine the step-size to use for forward-mode finite differences.

# Arguments

* `::ForwardMode`: An instance of the `ForwardMode` type.
* `x::AbstractFloat`: A point at which the derivative of a function will be
    approximated. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite differences at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `sqrt(eps(typeof(x)))`.

# Examples

```julia
import FiniteDiff: step_size, ForwardMode
ϵ = step_size(ForwardMode(), 0.0)
```
"""
@inline function step_size(::ForwardMode, x::AbstractFloat)::Real
    ϵ = sqrt(eps(typeof(x))) * abs(x)
    x_pe = x + ϵ
    return x_pe - x
end

"""
    step_size(::BackwardMode, x::AbstractFloat)

# Description

Determine the step-size to use for backward-mode finite differences.

# Arguments

* `::BackwardMode`: An instance of the `BackwardMode` type.
* `x::AbstractFloat`: A point at which the derivative of a function will be
    approximated. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite differences at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `sqrt(eps(typeof(x)))`.

# Examples

```julia
import FiniteDiff: step_size, BackwardMode
ϵ = step_size(BackwardMode(), 0.0)
```
"""
@inline function step_size(::BackwardMode, x::AbstractFloat)::Real
    ϵ = sqrt(eps(typeof(x))) * abs(x)
    x_pe = x + ϵ
    return x_pe - x
end

"""
    step_size(::CentralMode, x::AbstractFloat)

# Description

Determine the step-size to use for central-mode finite differences.

# Arguments

* `::CentralMode`: An instance of the `CentralMode` type.
* `x::AbstractFloat`: A point at which the derivative of a function will be
    approximated. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite differences at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `cbrt(eps(typeof(x)))`.

# Examples

```julia
import FiniteDiff: step_size, CentralMode
ϵ = step_size(CentralMode(), 0.0)
```
"""
@inline function step_size(::CentralMode, x::AbstractFloat)::Real
    ϵ = cbrt(eps(typeof(x))) * abs(x)
    x_pe = x + ϵ
    return x_pe - x
end

"""
    step_size(::ComplexMode, x::AbstractFloat)

# Description

Determine the step-size to use for complex-mode finite differences.

# Arguments

* `::ComplexMode`: An instance of the `ComplexMode` type.
* `x::AbstractFloat`: A point at which the derivative of a function will be
    approximated. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite differences at `x`.

# References

See "The Complex-Step Derivative Approximation" by Martins, Sturdza and Alonso
(2003) for the mathematical justification for working with `eps(x)`.

# Examples

```julia
import FiniteDiff: step_size, ComplexMode
ϵ = step_size(ComplexMode(), 0.0)
```
"""
@inline function step_size(::ComplexMode, x::AbstractFloat)::Real
    ϵ = eps(typeof(x))
    x_pe = x + ϵ
    return x_pe - x
end

"""
    step_size(::HessianMode, x::AbstractFloat)

# Description

Determine the step-size to use for finite differences of hessians.

# Arguments

* `::HessianMode`: An instance of the `HessianMode` type.
* `x::AbstractFloat`: A point at which the derivative of a function will be
    approximated. Its type must implement `eps`.

# Returns

* `ϵ::Real`: The step-size to use for finite differences at `x`.

# References

See Section 5.7 of Numerical Recipes in C for the mathematical justification
for working with `eps(x)^(1 // 4)`.

# Examples

```julia
import FiniteDiff: step_size, HessianMode
ϵ = step_size(HessianMode(), 0.0)
```
"""
@inline function step_size(::HessianMode, x::AbstractFloat)::Real
    ϵ = eps(typeof(x))^(1 // 4) * abs(x)
    x_pe = x + ϵ
    return x_pe - x
end