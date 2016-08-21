# Description

```@meta
CurrentModule = FiniteDiff
```

The `second_derivative` function is used to compute the second derivative of a
univariate function that maps $\mathbb{R}$ to $\mathbb{R}$. For example,
`x -> sin(x)` is a univariate function that might be passed to the
`second_derivative` function.

The `second_derivative` function comes in three core variants:

* A pure function that directly computes the second derivative of a function
    `f` at a fixed numeric value of `x` and returns a number as a result.
* A mutating function that computes the second derivative of a function `f` at
    a fixed numeric value of `x` and stores the result into a user-provided
    output array. This function returns `nothing` since its action is focused
    on mutation.
* A higher-order function that constructs a new function `f_prime2` that will
    compute the second derivative of `f` at any point `x` that is provided as
    an argument to `f_prime2`. This newly constructed function is the return
    value for this variant of the `second_derivative` function.

# Primary Methods

The primary methods that most users will want to use are the following:

* Use `second_derivative(f::Function, x::AbstractFloat)` to approximate the
    second derivative of `f` at `x`. This is the pure variant described above.
* Use `second_derivative!(output::AbstractArray, f::Function, x::AbstractFloat)`
    to approximate the second derivative of `f` at `x` and store the result
    into the `output` array.
* Use `second_derivative(f::Function)` to generate a new function that
    approximates the true second derivative function of `f`. The new function
    `f_prime2` can be evaluated at any point `x` that is desired after it is
    constructed.

# Detailed Method-Level Documentation

```@docs
second_derivative(f::Function, x::AbstractFloat)
```

```@docs
second_derivative!(
    output::AbstractArray,
    f::Function,
    x::AbstractFloat,
)
```

```@docs
second_derivative(f::Function; mutates::Bool=false)
```
