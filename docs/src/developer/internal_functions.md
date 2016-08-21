# Overview

```@meta
CurrentModule = FiniteDiff
```

We use dispatch to determine the mode of finite difference we use to perform
numerical differentiation. The modes are specified using the following types:

* `ForwardMode`
* `BackwardMode`
* `CentralMode`
* `ComplexMode`
* `HessianMode`

Given a mode and a value of `x` at which to differentiate `f(x)`, the following
functions are used to determine the step-size to use for finite difference
approximations of derivatives:

* `step_size(::ForwardMode, x::AbstractFloat)`
* `step_size(::BackwardMode, x::AbstractFloat)`
* `step_size(::CentralMode, x::AbstractFloat)`
* `step_size(::ComplexMode, x::AbstractFloat)`
* `step_size(::HessianMode, x::AbstractFloat)`

Unless there are bugs in the definitions of the core functions, the quality of
an approximate derivative is almost entirely determined by these step-sizes.
These step sizes are chosen to reach a reasonable compromise between errors
introduced by floating-point truncation (which could be minimized by using
larger step-sizes) and errors introduced by the use of an truncated Taylor
series (which could be minimized by using smaller step-sizes). The exact
compromise depends on the mode of finite differences we use.

# Detailed Method-Level Documentation

```@docs
step_size(::ForwardMode, x::AbstractFloat)
```

---

```@docs
step_size(::BackwardMode, x::AbstractFloat)
```

---

```@docs
step_size(::CentralMode, x::AbstractFloat)
```

---

```@docs
step_size(::ComplexMode, x::AbstractFloat)
```

---

```@docs
step_size(::HessianMode, x::AbstractFloat)
```
