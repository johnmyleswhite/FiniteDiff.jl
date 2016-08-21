# Description

```@meta
CurrentModule = FiniteDiff
```

The `derivative` function is used to compute the derivative of a univariate
function that maps $\mathbb{R}$ to $\mathbb{R}$. For example, `x -> sin(x)` is
a univariate function that might be passed to the `derivative` function.

The `derivative` function comes in three core variants:

* A pure function that directly computes the derivative of a function `f` at a
    fixed numeric value of `x` and returns a number as a result.
* A mutating function that computes the derivative of a function `f` at a fixed
    numeric value of `x` and stores the result into a user-provided output
    array. This function returns `nothing` since its action is focused on
    mutation.
* A higher-order function that constructs a new function `f_prime` that will
    compute the derivative of `f` at any point `x` that is provided as an
    argument to `f_prime`. This newly constructed function is the return value
    for this variant of the `derivative` function.

In addition, the `derivative` function allows one to chose the mode of
numerical differentiation that is used to compute an approximate derivative.
The default is use central finite differences, which evaluates `f` at two
nearby points and provides much higher accuracy than other strategies at a cost
of roughly double the amount of computation. Other modes of numerical
differentiation are available, but must be opted-in to explicitly.

# Primary Methods

The primary methods that most users will want to use are the following:

* Use `derivative(f::Function, x::AbstractFloat)` to approximate the derivative
    of `f` at `x`. This is the pure variant described above. Calling this
    function is equivalent to calling
    `derivative(f::Function, x::AbstractFloat, ::CentralMode)`, but leaves the
    selection of the finite difference mode up to the package author's
    discretion.
* Use `derivative!(output::AbstractArray, f::Function, x::AbstractFloat)` to
    approximate the derivative of `f` at `x` and store the result into the
    `output` array.
* Use `derivative(f::Function)` to generate a new function that approximates
    the true derivative function of `f`. The new function `f_prime` can be
    evaluated at any point `x` that is desired after it is constructed.

# Detailed Method-Level Documentation

```@docs
derivative(f::Function, x::AbstractFloat)
```

```@docs
derivative!(
    output::AbstractArray,
    f::Function,
    x::AbstractFloat,
    m::Mode = CentralMode(),
)
```

```@docs
derivative(f::Function, mode::Mode = CentralMode(); mutates::Bool=false)
```

```@docs
derivative(f::Function, x::AbstractFloat, ::ForwardMode)
```

```@docs
derivative(f::Function, x::AbstractFloat, ::BackwardMode)
```

```@docs
derivative(f::Function, x::AbstractFloat, ::CentralMode)
```

```@docs
derivative(f::Function, x::AbstractFloat, ::ComplexMode)
```
