# Description

```@meta
CurrentModule = FiniteDiff
```

The `hessian` function is used to compute the Hessian of a multivariate
function that maps $\mathbb{R}^n$ to $\mathbb{R}$. For example,
`x -> sin(x[1]) + cos(x[2])` is a multivariate function that might be passed to
the `hessian` function.

The `hessian` function comes in three core variants:

* A pure function that directly computes the Hessian of a function `f` at a
    fixed value of `x` and returns a newly allocated array as a result.
* A mutating function that computes the Hessian of a function `f` at a fixed
    value of `x` and stores the result into a user-provided output array. This
    function returns `nothing` since its action is focused on mutation. To
    ensure thread safety, this mutating variant also requires that a buffer
    be provided that is part of the internal implementation of the function,
    but can be provided explicitly to minimize unnecessary memory allocations.
* A higher-order function that constructs a new function `f_prime` that will
    compute the Hessian of `f` at any point `x` that is provided as an
    argument to `f_prime`. This newly constructed function is the return value
    for this variant of the `hessian` function.

# Primary Methods

The primary methods that most users will want to use are the following:

* Use `hessian(f::Function, x::AbstractArray)` to approximate the Hessian
    of `f` at `x`. This is the pure variant described above.
* Use `hessian!(output::AbstractArray, f::Function, x::AbstractArray, buffer::AbstractArray)`
    to approximate the Hessian of `f` at `x` and store the result into the
    `output` array.
* Use `hessian(f::Function)` to generate a new function that approximates
    the true Hessian function of `f`. The new function `f_prime` can be
    evaluated at any point `x` that is desired after it is constructed.

# Detailed Method-Level Documentation

```@docs
hessian!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
)
```

---

```@docs
hessian{T <: AbstractFloat}(f::Function, x::AbstractArray{T})
```

---

```@docs
hessian(f::Function; mutates::Bool = false)
```
