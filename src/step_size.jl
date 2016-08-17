"""
Determine the step-size to use for forward-mode finite-differencing of
gradients.
"""
@inline function step_size(::ForwardMode, x::AbstractFloat)
    return sqrt(eps(x)) * max(one(x), abs(x))
end

"""
Determine the step-size to use for backward-mode finite-differencing of
gradients.
"""
@inline function step_size(::BackwardMode, x::AbstractFloat)
    return sqrt(eps(x)) * max(one(x), abs(x))
end

"""
Determine the step-size to use for central-mode finite-differencing of
gradients.
"""
@inline function step_size(::CentralMode, x::AbstractFloat)
    return cbrt(eps(x)) * max(one(x), abs(x))
end

"""
Determine the step-size to use for complex-mode finite-differencing of
gradients.
"""
@inline function step_size(::ComplexMode, x::AbstractFloat)
    return eps(x)
end

"""
Determine the step-size to use for finite-differencing of hessians.
"""
@inline function step_size(::HessianMode, x::AbstractFloat)
    return eps(x)^(1//4) * max(one(x), abs(x))
end
