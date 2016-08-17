"""
Evaluate the derivative of `f` at `x` using forward finite-differencing.
"""
@inline function derivative(f::Function, x::AbstractFloat, ::ForwardMode)
    ϵ = step_size(ForwardMode(), x)
    return (f(x + ϵ) - f(x)) / ϵ
end

"""
Evaluate the derivative of `f` at `x` using backward finite-differencing.
"""
@inline function derivative(f::Function, x::AbstractFloat, ::BackwardMode)
    ϵ = step_size(BackwardMode(), x)
    return (f(x) - f(x - ϵ)) / ϵ
end

"""
Evaluate the derivative of `f` at `x` using central finite-differencing.
"""
@inline function derivative(f::Function, x::AbstractFloat, ::CentralMode)
    ϵ = step_size(CentralMode(), x)
    return (f(x + ϵ) - f(x - ϵ)) / (ϵ + ϵ)
end

"""
Evaluate the derivative of `f` at `x` using complex finite-differencing.

NOTE: This will only work if the function `f` both (a) supports complex inputs
and (b) is analytic in the sense used in complex analysis.
"""
@inline function derivative(f::Function, x::AbstractFloat, ::ComplexMode)
    ϵ = step_size(ComplexMode(), x)
    return imag(f(x + ϵ * im)) / ϵ
end

"""
Evaluate the derivative of `f` at `x` using finite-differencing. Mutates
`output` to contain the result. See the documentation for the non-mutating
version for any additional information about the effects of choosing a specific
finite-differencing mode.
"""
function derivative!(
    output::AbstractArray,
    f::Function,
    x::AbstractFloat,
    m::Mode = CentralMode(),
)
    output[1] = derivative(f, x, m)
    return
end

"""
Evaluate the derivative of `f` at `x`. Defaults to using central
finite-differencing by calling `derivative(f, x, CentralMode())`.
"""
derivative(f::Function, x::AbstractFloat) = derivative(f, x, CentralMode())

"""
Construct a new function that will evaluate the derivative of `f` at any value
of `x`. The user can specify the mode as a positional argument and can also
indicate whether to return a mutating or non-mutating function using a keyword
argument.
"""
function derivative(f::Function, mode = CentralMode(); mutates::Bool=false)
    if mutates
        return (output, x) -> derivative!(output, f, x, mode)
    else
        return (x, ) -> derivative(f, x, mode)
    end
end
