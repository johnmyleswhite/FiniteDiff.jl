# Description

```@meta
CurrentModule = FiniteDiff
```

The `gradient` function is used to compute the derivative of a multivariate
function that maps $\mathbb{R}^n$ to $\mathbb{R}$. For example,
`x -> sin(x[1]) + cos(x[2])` is a multivariate function that might be passed to
the `gradient` function.

The `gradient` function comes in three core variants:

* A pure function that directly computes the gradient of a function `f` at a
    fixed value of `x` and returns a newly allocated array as a result.
* A mutating function that computes the gradient of a function `f` at a fixed
    value of `x` and stores the result into a user-provided output array. This
    function returns `nothing` since its action is focused on mutation. To
    ensure thread safety, this mutating variant also requires that a buffer
    be provided that is part of the internal implementation of the function,
    but can be provided explicitly to minimize unnecessary memory allocations.
* A higher-order function that constructs a new function `f_prime` that will
    compute the gradient of `f` at any point `x` that is provided as an
    argument to `f_prime`. This newly constructed function is the return value
    for this variant of the `gradient` function.

In addition, the `gradient` function allows one to chose the mode of
numerical differentiation that is used to compute an approximate derivative.
The default is use central finite differences, which evaluates `f` at two
nearby points for each dimension of the input array `x` and provides much
higher accuracy than other strategies at a cost of roughly double the amount of
computation. Other modes of numerical differentiation are available, but must
be opted-in to explicitly.

# Primary Methods

The primary methods that most users will want to use are the following:

* Use `gradient(f::Function, x::AbstractArray)` to approximate the gradient
    of `f` at `x`. This is the pure variant described above.
* Use `gradient!(output::AbstractArray, f::Function, x::AbstractArray, buffer::AbstractArray)`
    to approximate the gradient of `f` at `x` and store the result into the
    `output` array.
* Use `gradient(f::Function)` to generate a new function that approximates
    the true gradient function of `f`. The new function `f_prime` can be
    evaluated at any point `x` that is desired after it is constructed.

# Detailed Method-Level Documentation

```@docs
gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
)
```

---

```@docs
gradient{T <: AbstractFloat}(
    f::Function,
    x::AbstractArray{T},
    mode::Mode = CentralMode(),
)
```

---

```@docs
gradient(f::Function, mode::Mode = CentralMode(); mutates::Bool = false)
```

---

```@docs
gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
    ::ForwardMode,
)
```

---

```@docs
gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
    ::BackwardMode,
)
```

---

```@docs
gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{T},
    ::CentralMode,
)
```

---

```@docs
gradient!{T <: AbstractFloat}(
    output::AbstractArray,
    f::Function,
    x::AbstractArray{T},
    buffer::AbstractArray{Complex{T}},
    ::ComplexMode,
)
```
